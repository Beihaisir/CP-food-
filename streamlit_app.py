# -*- coding: utf-8 -*-
import re
import json
import tempfile
from pathlib import Path
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="餐饮日销售&菜品结构分析", layout="wide")


# -----------------------------
# Utilities
# -----------------------------
def _norm_col(s: str) -> str:
    return str(s).strip().replace("\u3000", " ")


def _to_dt_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _to_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


# -----------------------------
# Default configurable rules
# -----------------------------
DEFAULT_CHANNEL_RULES = {
    "抖音": ["(套)", "（套）", "(套餐)", "（套餐）"]
}

DEFAULT_TAG_RULES = {
    # 主原料/主题类（示例：你后续可以在侧边栏直接改/导入）
    "板筋类": ["板筋"],
    "猪肝类": ["猪肝"],
    "牛肉类": ["牛", "肥牛", "牛肉"],
    "鸡肉类": ["鸡", "鸡肉"],
    "猪肉类": ["猪", "五花", "里脊", "排骨"],
    "鱼虾类": ["鱼", "虾", "蟹", "贝"],
    "豆制品类": ["豆腐", "豆皮", "千张", "豆干"],
    "蔬菜类": ["土豆", "青椒", "白菜", "菠菜", "藕", "金针菇", "木耳", "海带"],
    "主食类": ["米饭", "面", "粉", "馍", "饼", "饭"],
    "饮品类": ["可乐", "雪碧", "茶", "奶茶", "果汁", "饮料", "水"],
}

DEFAULT_SPEC_RULES = {
    # 将“规格名称”规范成两个正交维度：基底/主食 & 份量
    "base": {
        "宽面": [r"宽面"],
        "细面": [r"细面"],
        "宽粉": [r"宽粉"],
        "米饭": [r"米饭|白饭|米饭\*?\d*"],
        "粉":   [r"粉(?!丝)"],
        "面":   [r"面"],
        "无主食/未知": [r".*"],  # 兜底
    },
    "size": {
        "标准": [r"标准|中份|普通"],
        "大份": [r"大份|加大|大碗|大"],
        "小份": [r"小份|小碗|小"],
        "未知": [r".*"],
    }
}

# 用于“自动表头识别”的关键字段
SALES_KEYS = {"日期", "含税销售额", "去税销售额", "客流量", "客单", "门店名称", "门店代码", "销售数量"}
ITEMS_KEYS = {"创建时间", "菜品名称", "菜品数量", "POS销售单号", "单据类型", "规格名称", "优惠后小计价格", "小计价格"}


# -----------------------------
# Smart Excel loader (auto header row)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_excel_smart(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    智能读取 Excel：自动识别真正表头行（解决：导出文件前几行是标题/导出信息/空行等）
    """
    try:
        preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=40)
        preview = preview.replace({np.nan: ""})

        best_row = None
        best_score = -1

        for i in range(len(preview)):
            row_vals = set(str(x).strip() for x in preview.iloc[i].tolist() if str(x).strip() != "")
            score = len(row_vals & SALES_KEYS) + len(row_vals & ITEMS_KEYS)
            if score > best_score:
                best_score = score
                best_row = i

        # 至少命中 2 个关键字段才认为找到了表头
        if best_row is not None and best_score >= 2:
            df = pd.read_excel(path, sheet_name=sheet_name, header=best_row)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name)

        if isinstance(df, dict):
            df = list(df.values())[0]

        df.columns = [_norm_col(c) for c in df.columns]
        return df
    except Exception as e:
        raise RuntimeError(f"读取失败：{path}\n{e}")


# -----------------------------
# Rules: channel / tags / spec
# -----------------------------
def detect_channel(dish_name: str, channel_rules: Dict[str, List[str]]) -> str:
    if _is_nan(dish_name):
        return "未知"
    name = str(dish_name)
    for ch, keys in channel_rules.items():
        if any(k in name for k in keys):
            return ch
    return "非抖音/未知"


def extract_tags(dish_name: str, tag_rules: Dict[str, List[str]]) -> List[str]:
    """
    多标签：一个菜可命中多个类；计数时 explode 后每个类各计一次（不分摊、不稀释）。
    """
    if _is_nan(dish_name):
        return ["未分类"]
    name = str(dish_name)
    tags = []
    for tag, keys in tag_rules.items():
        if any(k in name for k in keys):
            tags.append(tag)
    return tags if tags else ["未分类"]


def normalize_spec(spec_name: str, spec_rules: Dict[str, Dict[str, List[str]]]) -> Tuple[str, str]:
    """
    规格 -> base/size 两维
    """
    spec = "" if _is_nan(spec_name) else str(spec_name)

    base_map = spec_rules.get("base", {})
    size_map = spec_rules.get("size", {})

    base = "无主食/未知"
    for k, pats in base_map.items():
        if any(re.search(p, spec) for p in pats):
            base = k
            break

    size = "未知"
    for k, pats in size_map.items():
        if any(re.search(p, spec) for p in pats):
            size = k
            break

    return base, size


# -----------------------------
# Prepare: items (large-friendly)
# -----------------------------
@st.cache_data(show_spinner=False)
def prepare_items(
    items: pd.DataFrame,
    channel_rules: Dict[str, List[str]],
    tag_rules: Dict[str, List[str]],
    spec_rules: Dict[str, Dict[str, List[str]]],
    include_refund: bool,
    only_normal_status: bool,
    time_col_hint: Optional[str] = None,
    order_col_hint: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    df = items.copy()
    df.columns = [_norm_col(c) for c in df.columns]

    # Best-effort column mapping (允许 hint 覆盖)
    col_time = time_col_hint if (time_col_hint and time_col_hint in df.columns) else ("创建时间" if "创建时间" in df.columns else None)
    col_dish = "菜品名称" if "菜品名称" in df.columns else None
    col_qty = "菜品数量" if "菜品数量" in df.columns else None
    col_spec = "规格名称" if "规格名称" in df.columns else None
    col_status = "菜品状态" if "菜品状态" in df.columns else None
    col_order = order_col_hint if (order_col_hint and order_col_hint in df.columns) else ("POS销售单号" if "POS销售单号" in df.columns else None)
    col_doc_type = "单据类型" if "单据类型" in df.columns else None
    col_refund = "POS退款单号" if "POS退款单号" in df.columns else None

    # amount columns (prefer discounted)
    col_amt = None
    for c in ["优惠后小计价格", "小计价格"]:
        if c in df.columns:
            col_amt = c
            break

    missing = []
    for k, v in [("创建时间", col_time), ("菜品名称", col_dish), ("菜品数量", col_qty), ("POS销售单号", col_order), ("金额列(优惠后小计价格/小计价格)", col_amt)]:
        if v is None:
            missing.append(k)
    if missing:
        raise RuntimeError(f"菜品明细缺少关键字段：{missing}\n当前列名：{list(df.columns)}")

    df[col_time] = _to_dt_series(df[col_time])
    df = df.dropna(subset=[col_time]).copy()

    df["date"] = df[col_time].dt.date
    df["qty"] = _to_num_series(df[col_qty]).fillna(0.0)
    df["amount"] = _to_num_series(df[col_amt]).fillna(0.0)

    # Refund handling
    if not include_refund:
        refund_mask = pd.Series(False, index=df.index)
        if col_doc_type is not None:
            refund_mask |= df[col_doc_type].astype(str).str.contains("退|退款|红冲|冲正", na=False)
        if col_refund is not None:
            refund_mask |= df[col_refund].notna() & (df[col_refund].astype(str).str.len() > 0)
        df = df.loc[~refund_mask].copy()

    # Status filter
    if only_normal_status and col_status is not None:
        df = df.loc[df[col_status].astype(str).str.contains("正常", na=False)].copy()

    # Enrich: channel, spec, tags
    df["channel"] = df[col_dish].apply(lambda x: detect_channel(x, channel_rules))
    if col_spec is not None:
        df["spec_base"], df["spec_size"] = zip(*df[col_spec].apply(lambda x: normalize_spec(x, spec_rules)))
    else:
        df["spec_base"] = "无主食/未知"
        df["spec_size"] = "未知"

    df["tags"] = df[col_dish].apply(lambda x: extract_tags(x, tag_rules))
    df_tags = df.explode("tags").rename(columns={"tags": "tag"}).copy()

    # Order-level (for decomposition)
    order = (df.groupby(["date", col_order], as_index=False)
               .agg(order_amount=("amount", "sum"),
                    order_qty=("qty", "sum"),
                    order_lines=(col_order, "size")))

    # Daily overall from items
    daily = (df.groupby("date", as_index=False)
               .agg(items_amount=("amount", "sum"),
                    items_qty=("qty", "sum"),
                    items_orders=(col_order, "nunique"),
                    items_lines=(col_order, "size")))

    # Daily by channel
    daily_ch = (df.groupby(["date", "channel"], as_index=False)
                  .agg(amount=("amount", "sum"),
                       qty=("qty", "sum"),
                       orders=(col_order, "nunique")))

    # Daily by tag (multi-count)
    daily_tag = (df_tags.groupby(["date", "tag"], as_index=False)
                    .agg(amount=("amount", "sum"),
                         qty=("qty", "sum"),
                         orders=(col_order, "nunique")))

    # Daily by spec
    daily_spec = (df.groupby(["date", "spec_base", "spec_size"], as_index=False)
                    .agg(amount=("amount", "sum"),
                         qty=("qty", "sum"),
                         orders=(col_order, "nunique")))

    return {
        "items_raw": df,
        "order": order,
        "daily": daily,
        "daily_channel": daily_ch,
        "daily_tag": daily_tag,
        "daily_spec": daily_spec
    }


# -----------------------------
# Prepare: sales (robust date column)
# -----------------------------
@st.cache_data(show_spinner=False)
def prepare_sales(sales: pd.DataFrame, date_col_hint: Optional[str] = None) -> pd.DataFrame:
    df = sales.copy()
    df.columns = [_norm_col(c) for c in df.columns]

    # If user specified date column, use it
    if date_col_hint and date_col_hint in df.columns:
        date_col = date_col_hint
    else:
        # Candidate columns by name keywords
        keys = ["日期", "日 期", "营业日期", "交易日期", "统计日期", "业务日期", "biz_date", "business_date", "order_date", "date"]
        candidates = []
        for c in df.columns:
            cl = str(c).lower()
            if any(k.lower() in cl for k in keys):
                candidates.append(c)

        if not candidates:
            candidates = list(df.columns)

        # Content-based scoring: parse success rate
        best_col, best_score, best_parsed = None, -1.0, None
        for c in candidates:
            s = _to_dt_series(df[c])
            score = s.notna().mean()
            if score > 0:
                uniq = s.dropna().dt.date.nunique()
                if uniq > 1:
                    score += 0.05
            if score > best_score:
                best_score, best_col, best_parsed = score, c, s

        if best_col is None or best_score < 0.30:
            raise RuntimeError(
                "日销售表无法识别日期列。\n"
                f"已读取列名：{list(df.columns)}\n"
                "请在左侧选择正确的日期列名。"
            )
        date_col = best_col
        df[date_col] = best_parsed

    df[date_col] = _to_dt_series(df[date_col])
    df = df.dropna(subset=[date_col]).copy()
    df["date"] = df[date_col].dt.date

    # Numeric best-effort
    for c in ["含税销售额", "去税销售额", "销售数量", "客流量", "客单"]:
        if c in df.columns:
            df[c] = _to_num_series(df[c])

    iso = df[date_col].dt.isocalendar()
    df["weekday"] = df[date_col].dt.weekday
    df["weekday_name"] = df[date_col].dt.day_name()
    df["iso_year"] = iso["year"].astype("Int64")
    df["iso_week"] = iso["week"].astype("Int64")
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)
    return df


# -----------------------------
# Analytics helpers
# -----------------------------
def overlap_range(a: pd.Series, b: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    a0, a1 = pd.to_datetime(a).min(), pd.to_datetime(a).max()
    b0, b1 = pd.to_datetime(b).min(), pd.to_datetime(b).max()
    lo, hi = max(a0, b0), min(a1, b1)
    if pd.isna(lo) or pd.isna(hi) or lo > hi:
        return None, None
    return lo, hi


def add_rolling_anomaly(df: pd.DataFrame, value_col: str, window: int = 28) -> pd.DataFrame:
    d = df.sort_values("date").copy()
    x = d[value_col].astype(float)
    mu = x.rolling(window, min_periods=max(7, window // 4)).mean()
    sd = x.rolling(window, min_periods=max(7, window // 4)).std(ddof=0)
    z = (x - mu) / sd.replace(0, np.nan)
    d["zscore"] = z
    d["is_anomaly"] = d["zscore"].abs() >= 2.0
    return d


def decompose_change(daily_items: pd.DataFrame) -> pd.DataFrame:
    d = daily_items.sort_values("date").copy()
    d["qty_per_order"] = d["items_qty"] / d["items_orders"].replace(0, np.nan)
    d["price_per_item"] = d["items_amount"] / d["items_qty"].replace(0, np.nan)

    roll_orders = d["items_orders"].shift(1).rolling(7, min_periods=3).mean()
    roll_qpo = d["qty_per_order"].shift(1).rolling(7, min_periods=3).mean()
    roll_ppi = d["price_per_item"].shift(1).rolling(7, min_periods=3).mean()

    d["base_orders"] = roll_orders
    d["base_qpo"] = roll_qpo
    d["base_ppi"] = roll_ppi
    d["base_amount"] = d["base_orders"] * d["base_qpo"] * d["base_ppi"]
    d["chg_vs_base"] = (d["items_amount"] - d["base_amount"])
    return d


def week_shape(df_sales: pd.DataFrame, value_col: str = "含税销售额") -> pd.DataFrame:
    d = df_sales.copy()
    d = d[d["weekday"].between(0, 4)].copy()
    wk = (d.groupby(["iso_year", "iso_week", "weekday"], as_index=False)
            .agg(value=(value_col, "sum")))
    wk["week_total"] = wk.groupby(["iso_year", "iso_week"])["value"].transform("sum")
    wk["share"] = wk["value"] / wk["week_total"].replace(0, np.nan)
    wk["weekday_name"] = wk["weekday"].map({0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"})
    return wk


def delta_contribution(
    daily_dim: pd.DataFrame,
    dim_col: str,
    amount_col: str,
    period_a: Tuple[date, date],
    period_b: Tuple[date, date],
    top_n: int = 20
) -> pd.DataFrame:
    a0, a1 = period_a
    b0, b1 = period_b
    d = daily_dim.copy()
    A = d[(d["date"] >= a0) & (d["date"] <= a1)].groupby(dim_col)[amount_col].sum()
    B = d[(d["date"] >= b0) & (d["date"] <= b1)].groupby(dim_col)[amount_col].sum()
    out = pd.DataFrame({"A": A, "B": B}).fillna(0.0)
    out["delta"] = out["B"] - out["A"]
    out = out.sort_values("delta")
    worst = out.head(top_n // 2)
    best = out.tail(top_n - len(worst))
    out2 = pd.concat([worst, best]).reset_index().rename(columns={dim_col: "维度"})
    return out2.sort_values("delta")


# -----------------------------
# Sidebar: upload & rules
# -----------------------------
st.sidebar.header("数据与口径（上传文件）")

sales_up = st.sidebar.file_uploader("上传日销售报表（.xls/.xlsx）", type=["xls", "xlsx"])
items_up = st.sidebar.file_uploader("上传订单菜品报告（.xls/.xlsx）", type=["xls", "xlsx"])


def _save_upload(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


sales_path = _save_upload(sales_up)
items_path = _save_upload(items_up)

include_refund = st.sidebar.checkbox("包含退款/红冲", value=False)
only_normal_status = st.sidebar.checkbox("仅统计“正常”菜品状态（若有字段）", value=False)

st.sidebar.divider()
st.sidebar.subheader("规则配置（可直接改JSON）")


def parse_json(text: str, fallback):
    try:
        return json.loads(text)
    except Exception:
        st.sidebar.error("JSON解析失败：将使用默认规则。请检查逗号/引号是否正确。")
        return fallback


channel_rules_text = st.sidebar.text_area(
    "渠道识别规则 channel_rules（菜品名称包含关键词即命中）",
    value=json.dumps(DEFAULT_CHANNEL_RULES, ensure_ascii=False, indent=2),
    height=120
)
tag_rules_text = st.sidebar.text_area(
    "菜品多标签规则 tag_rules（一个菜可命中多个类；计数各计一次）",
    value=json.dumps(DEFAULT_TAG_RULES, ensure_ascii=False, indent=2),
    height=220
)
spec_rules_text = st.sidebar.text_area(
    "规格规则 spec_rules（base/size 两条维度）",
    value=json.dumps(DEFAULT_SPEC_RULES, ensure_ascii=False, indent=2),
    height=240
)

channel_rules = parse_json(channel_rules_text, DEFAULT_CHANNEL_RULES)
tag_rules = parse_json(tag_rules_text, DEFAULT_TAG_RULES)
spec_rules = parse_json(spec_rules_text, DEFAULT_SPEC_RULES)

# Require uploads
st.title("餐饮经营分析（日销售 × 订单菜品明细）")
if sales_path is None or items_path is None:
    st.info("请在左侧先上传两个文件（日销售报表、订单菜品报告）后开始分析。")
    st.stop()

# -----------------------------
# Load data (smart header)
# -----------------------------
with st.spinner("读取数据（智能识别表头）..."):
    sales_raw = load_excel_smart(sales_path)
    items_raw = load_excel_smart(items_path)

# Sidebar: allow manual mapping for date/time/order fields
st.sidebar.divider()
st.sidebar.subheader("字段映射（识别不准时手动选）")

sales_date_col = st.sidebar.selectbox(
    "日销售：日期列",
    options=["(自动识别)"] + list(sales_raw.columns),
    index=0
)
sales_date_hint = None if sales_date_col == "(自动识别)" else sales_date_col

items_time_col = st.sidebar.selectbox(
    "菜品明细：创建时间列",
    options=["(自动识别)"] + list(items_raw.columns),
    index=0
)
items_time_hint = None if items_time_col == "(自动识别)" else items_time_col

items_order_col = st.sidebar.selectbox(
    "菜品明细：POS销售单号列",
    options=["(自动识别)"] + list(items_raw.columns),
    index=0
)
items_order_hint = None if items_order_col == "(自动识别)" else items_order_col

with st.spinner("清洗与预聚合..."):
    sales = prepare_sales(sales_raw, date_col_hint=sales_date_hint)
    items_pack = prepare_items(
        items_raw,
        channel_rules=channel_rules,
        tag_rules=tag_rules,
        spec_rules=spec_rules,
        include_refund=include_refund,
        only_normal_status=only_normal_status,
        time_col_hint=items_time_hint,
        order_col_hint=items_order_hint
    )

items_daily = items_pack["daily"]
daily_channel = items_pack["daily_channel"]
daily_tag = items_pack["daily_tag"]
daily_spec = items_pack["daily_spec"]

# -----------------------------
# Global filters
# -----------------------------
# Store selector if present
if "门店名称" in sales.columns:
    stores = ["全部"] + sorted(sales["门店名称"].dropna().astype(str).unique().tolist())
    store_choice = st.selectbox("门店（来自日销售表）", stores, index=0)
    if store_choice != "全部":
        sales_f = sales[sales["门店名称"].astype(str) == store_choice].copy()
    else:
        sales_f = sales.copy()
else:
    sales_f = sales.copy()

# Date range: default to overlap
sales_dates = pd.to_datetime(sales_f["date"])
items_dates = pd.to_datetime(items_daily["date"])
lo, hi = overlap_range(sales_dates, items_dates)
if lo is None:
    default_start = sales_dates.min().date()
    default_end = sales_dates.max().date()
    st.warning("两表没有重叠日期区间：将使用日销售表日期范围展示；关联分析将受限。")
else:
    default_start = lo.date()
    default_end = hi.date()

date_range = st.date_input("日期范围（默认取两表交集）", value=(default_start, default_end))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, default_end

sales_f = sales_f[(pd.to_datetime(sales_f["date"]).dt.date >= start_date) &
                  (pd.to_datetime(sales_f["date"]).dt.date <= end_date)].copy()
items_daily_f = items_daily[(items_daily["date"] >= start_date) &
                            (items_daily["date"] <= end_date)].copy()

metric = st.selectbox("核心指标（来自日销售表）", ["含税销售额", "客流量", "客单", "销售数量"], index=0)
if metric not in sales_f.columns:
    st.warning(f"日销售表缺少字段：{metric}。将回退使用“含税销售额”或任意数值列。")
    if "含税销售额" in sales_f.columns:
        metric = "含税销售额"
    else:
        num_cols = sales_f.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("日销售表没有可用的数值指标列，请检查数据。")
            st.stop()
        metric = num_cols[0]

# -----------------------------
# Tabs (decision-path)
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 发生了什么（趋势&异常）",
    "2 为什么（订单×件数×价格）",
    "3 周节奏（周一~周五）",
    "4 谁在驱动（类/规格/渠道）",
    "5 异常日钻取（闭环）"
])

# ---------- Tab 1: Trend & anomaly ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    if "含税销售额" in sales_f.columns:
        c1.metric("含税销售额（选定范围）", f"{float(sales_f['含税销售额'].sum(skipna=True)):,.0f}")
    else:
        c1.metric(f"{metric}（选定范围）", f"{float(sales_f[metric].sum(skipna=True)):,.0f}")

    if "客流量" in sales_f.columns:
        c2.metric("客流量（选定范围）", f"{float(sales_f['客流量'].sum(skipna=True)):,.0f}")
    else:
        c2.metric("客流量（选定范围）", "—")

    if "客单" in sales_f.columns:
        c3.metric("平均客单", f"{float(sales_f['客单'].mean(skipna=True)):,.2f}")
    else:
        c3.metric("平均客单", "—")

    c4.metric("菜品明细汇总金额", f"{float(items_daily_f['items_amount'].sum()) if len(items_daily_f) else 0.0:,.0f}")

    left, right = st.columns([2, 1])

    with left:
        trend = sales_f.groupby("date", as_index=False)[metric].sum().sort_values("date")
        fig = px.line(trend, x="date", y=metric, markers=False, title=f"{metric} 日趋势")
        st.plotly_chart(fig, use_container_width=True)

        if len(trend) >= 10:
            an = add_rolling_anomaly(trend, metric, window=28)
            fig2 = px.scatter(an, x="date", y=metric, color="is_anomaly",
                              hover_data=["zscore"], title=f"{metric} 异常识别（滚动zscore）")
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        if "iso_week" in sales_f.columns:
            pivot = (sales_f.groupby(["iso_year", "iso_week", "weekday"], as_index=False)[metric].sum())
            pivot = pivot[pivot["weekday"].between(0, 6)]
            pivot["weekday_name"] = pivot["weekday"].map({0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"})
            hm = pivot.pivot_table(index="iso_week", columns="weekday_name", values=metric, aggfunc="sum")
            fig_hm = px.imshow(hm, aspect="auto", title="周-星期 热力图（越亮=越高）")
            st.plotly_chart(fig_hm, use_container_width=True)

        st.subheader("两表对齐检查（口径差异）")
        merged = pd.merge(
            trend.rename(columns={metric: "sales_metric"}),
            items_daily_f.rename(columns={"items_amount": "items_metric"}),
            on="date", how="inner"
        )
        if len(merged):
            merged["diff"] = merged["sales_metric"] - merged["items_metric"]
            merged["diff_pct"] = merged["diff"] / merged["sales_metric"].replace(0, np.nan)
            st.dataframe(
                merged.sort_values("diff_pct", key=lambda s: s.abs(), ascending=False).head(10),
                use_container_width=True
            )
        else:
            st.info("当前日期范围内无法做对齐检查（无交集或明细为空）。")

# ---------- Tab 2: Decomposition ----------
with tab2:
    st.subheader("销售变化拆解（订单数 × 每单件数 × 平均成交价/件）")
    if len(items_daily_f) < 5:
        st.info("明细日汇总太少，无法稳定拆解。请扩大日期范围或确认明细数据。")
    else:
        dec = decompose_change(items_daily_f)
        dec = dec[(dec["date"] >= start_date) & (dec["date"] <= end_date)].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dec["date"], y=dec["items_amount"], name="明细金额（当日）"))
        fig.add_trace(go.Scatter(x=dec["date"], y=dec["base_amount"], name="7日基线金额（不含当日）"))
        fig.update_layout(title="明细金额 vs 7日基线", xaxis_title="日期", yaxis_title="金额")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("平均订单数", f"{dec['items_orders'].mean():,.1f}")
        c2.metric("平均每单件数", f"{(dec['items_qty']/dec['items_orders'].replace(0,np.nan)).mean():,.2f}")
        c3.metric("平均成交价/件", f"{(dec['items_amount']/dec['items_qty'].replace(0,np.nan)).mean():,.2f}")

        show_n = st.slider("显示偏离基线Top N天", 5, 30, 10)
        key = dec.dropna(subset=["chg_vs_base"]).copy()
        key["abs_dev"] = key["chg_vs_base"].abs()
        st.dataframe(
            key.sort_values("abs_dev", ascending=False).head(show_n)[
                ["date", "items_amount", "base_amount", "chg_vs_base", "items_orders"]
            ],
            use_container_width=True
        )

# ---------- Tab 3: Week rhythm ----------
with tab3:
    st.subheader("周一~周五：周节奏健康度（形状变化比高低更重要）")
    if metric not in sales_f.columns or len(sales_f) < 10:
        st.info("日销售数据不足。")
    else:
        wk = week_shape(sales_f, value_col=metric)

        fig = px.line(
            wk, x="weekday_name", y="share",
            color=wk["iso_year"].astype(str) + "-W" + wk["iso_week"].astype(str),
            markers=True,
            title="每周（周一~周五）占比形状（share=当周weekday金额/周一~周五总金额）"
        )
        fig.update_layout(xaxis_title="星期", yaxis_title="占比")
        st.plotly_chart(fig, use_container_width=True)

        avg = wk.groupby("weekday")["share"].mean().reset_index()
        avg["weekday_name"] = avg["weekday"].map({0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"})
        fig2 = px.bar(avg, x="weekday_name", y="share", title="历史平均周内形状（周一~周五占比均值）")
        st.plotly_chart(fig2, use_container_width=True)

        stab = wk.groupby("weekday")["share"].agg(["mean", "std"]).reset_index()
        stab["cv"] = stab["std"] / stab["mean"].replace(0, np.nan)
        stab["weekday_name"] = stab["weekday"].map({0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"})
        st.dataframe(stab.sort_values("cv", ascending=False)[["weekday_name", "mean", "std", "cv"]], use_container_width=True)

# ---------- Tab 4: Drivers ----------
with tab4:
    st.subheader("驱动因素：类（多标签计数）/ 规格 / 渠道（抖音套餐）")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 渠道：抖音 vs 非抖音/未知")
        dch = daily_channel[(daily_channel["date"] >= start_date) & (daily_channel["date"] <= end_date)].copy()
        if len(dch):
            fig = px.area(dch, x="date", y="amount", color="channel", title="渠道金额（日）")
            st.plotly_chart(fig, use_container_width=True)
            ch_sum = dch.groupby("channel", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            st.dataframe(ch_sum, use_container_width=True)
        else:
            st.info("无渠道数据。")

        st.markdown("### 规格：基底×份量")
        dsp = daily_spec[(daily_spec["date"] >= start_date) & (daily_spec["date"] <= end_date)].copy()
        if len(dsp):
            base_sum = dsp.groupby("spec_base", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(12)
            figb = px.bar(base_sum, x="spec_base", y="amount", title="主食/基底规格金额 Top")
            st.plotly_chart(figb, use_container_width=True)

            size_sum = dsp.groupby("spec_size", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
            figs = px.pie(size_sum, names="spec_size", values="amount", title="份量规格金额占比")
            st.plotly_chart(figs, use_container_width=True)
        else:
            st.info("无规格数据。")

    with c2:
        st.markdown("### 类别（多标签计数：一菜可同时计入多个类）")
        dtg = daily_tag[(daily_tag["date"] >= start_date) & (daily_tag["date"] <= end_date)].copy()
        if len(dtg):
            topn = st.slider("Top N 类别", 5, 30, 15)
            tag_sum = dtg.groupby("tag", as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(topn)
            fig = px.bar(tag_sum, x="tag", y="amount", title="类金额 Top（多标签累计）")
            st.plotly_chart(fig, use_container_width=True)

            st.caption("注意：这是“标签维度统计”，总和可能超过真实总销售额（因为一菜可属于多个类）。")

            st.markdown("### 类别变化贡献（两段时间对比）")
            mid = start_date + (end_date - start_date) / 2
            mid = pd.to_datetime(mid).date()
            pA = (start_date, mid)
            pB = (mid, end_date)
            st.write(f"默认对比：A={pA[0]}~{pA[1]} vs B={pB[0]}~{pB[1]}（可自行调整）")

            colA1, colA2, colB1, colB2 = st.columns(4)
            a0 = colA1.date_input("A开始", pA[0], key="a0")
            a1 = colA2.date_input("A结束", pA[1], key="a1")
            b0 = colB1.date_input("B开始", pB[0], key="b0")
            b1 = colB2.date_input("B结束", pB[1], key="b1")

            contrib = delta_contribution(dtg, "tag", "amount", (a0, a1), (b0, b1), top_n=20)
            figc = px.bar(contrib, x="维度", y="delta", title="类别贡献（B-A，负值=拖累，正值=拉动）")
            st.plotly_chart(figc, use_container_width=True)
            st.dataframe(contrib, use_container_width=True)
        else:
            st.info("无类别数据。")

# ---------- Tab 5: Drilldown ----------
with tab5:
    st.subheader("异常日钻取：先发现 → 再解释 → 再给动作对象（类/规格/渠道）")

    trend = sales_f.groupby("date", as_index=False)[metric].sum().sort_values("date")
    anomaly_days = []
    if len(trend) >= 10:
        an = add_rolling_anomaly(trend, metric, window=28)
        anomaly_days = an.loc[an["is_anomaly"], "date"].astype(str).tolist()
    if not anomaly_days and len(items_daily_f) >= 10:
        tmp = items_daily_f.rename(columns={"items_amount": "val"}).copy()
        iday = add_rolling_anomaly(tmp, "val", window=28)
        anomaly_days = iday.loc[iday["is_anomaly"], "date"].astype(str).tolist()

    if not anomaly_days:
        st.info("当前范围内未识别到明显异常日（或数据不足）。你仍然可以手动选择任意日期查看。")

    all_days = sorted(set(pd.to_datetime(sales_f["date"]).dt.date.astype(str).tolist()
                          + items_daily_f["date"].astype(str).tolist()))
    if not all_days:
        st.stop()

    default_day = anomaly_days[0] if anomaly_days else all_days[-1]
    day = st.selectbox("选择日期", all_days, index=all_days.index(default_day) if default_day in all_days else len(all_days) - 1)
    day_d = pd.to_datetime(day).date()

    srow = trend[trend["date"].astype(str) == day]
    irow = items_daily_f[items_daily_f["date"].astype(str) == day]

    c1, c2, c3 = st.columns(3)
    if len(srow):
        c1.metric(f"日销售 {metric}", f"{float(srow[metric].iloc[0]):,.0f}")
    if len(irow):
        c2.metric("明细金额", f"{float(irow['items_amount'].iloc[0]):,.0f}")
        c3.metric("明细订单数", f"{float(irow['items_orders'].iloc[0]):,.0f}")

    st.markdown("### 当日：渠道贡献")
    dch = daily_channel[daily_channel["date"] == day_d].sort_values("amount", ascending=False)
    st.dataframe(dch, use_container_width=True) if len(dch) else st.info("当日无渠道数据。")

    st.markdown("### 当日：类别贡献（多标签计数）")
    dtg = daily_tag[daily_tag["date"] == day_d].sort_values("amount", ascending=False).head(30)
    if len(dtg):
        fig = px.bar(dtg, x="tag", y="amount", title="当日类别金额Top（多标签累计）")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dtg, use_container_width=True)
    else:
        st.info("当日无类别数据。")

    st.markdown("### 当日：规格贡献（基底×份量）")
    dsp = daily_spec[daily_spec["date"] == day_d].sort_values("amount", ascending=False).head(30)
    if len(dsp):
        dsp2 = dsp.copy()
        dsp2["spec"] = dsp2["spec_base"].astype(str) + " / " + dsp2["spec_size"].astype(str)
        fig = px.bar(dsp2, x="spec", y="amount", title="当日规格金额Top")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dsp2[["spec_base", "spec_size", "amount", "qty", "orders"]], use_container_width=True)
    else:
        st.info("当日无规格数据。")

st.caption("说明：本应用优先做“大表可跑”的预聚合设计；规则（多标签、抖音识别、规格）均可在侧边栏用JSON调整。")
