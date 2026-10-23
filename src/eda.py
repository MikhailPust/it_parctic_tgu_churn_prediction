"""Этап 3: воспроизводимый разведочный анализ.

    python -m src.eda

Вход: data/prepared (результат python -m src.prepare) и data/raw/lesson3.
Выход: data/eda (исследовательские таблицы), reports/eda (рисунки, расчётные таблицы, проверки).
Анализ связей с оттоком выполняется только на датах EDA; исходы валидации и теста не рассчитываются.
"""
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, snapshots  # noqa: E402
from .clean import fmt_bool, fmt_date, fmt_money, fmt_ts  # noqa: E402
from .load import read_raw_csv, write_csv  # noqa: E402

# Палитра: категориальные цвета в фиксированном порядке, текст — нейтральными чернилами
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
REGION_COLOR = dict(zip(config.REGIONS, SERIES))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 150,
})

NUMERIC = ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub", "traffic_gb",
           "voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count", "support_count",
           "support_network_count", "support_billing_count", "support_unknown_topic_count"]
FLAGS = ["traffic_gb_missing", "voice_minutes_invalid"]


# ---------- загрузка подготовленных таблиц ----------

def load_prepared():
    P = config.PREPARED
    events = read_raw_csv(P / "contract_events_clean.csv")
    subs = read_raw_csv(P / "subscribers_clean.csv")
    subs["activation_date"] = pd.to_datetime(subs.activation_date)
    cm = read_raw_csv(P / "customer_month.csv")
    for c in NUMERIC:
        cm[c] = pd.to_numeric(cm[c].replace("", np.nan))
    for c in FLAGS:
        cm[c] = cm[c].eq("1")
    cm["billing_month"] = pd.to_datetime(cm.billing_month)
    cm["available_at"] = pd.to_datetime(cm.available_at, format="ISO8601").dt.tz_convert(config.TZ)
    return events, subs, cm


def load_network():
    n = read_raw_csv(config.SOURCES["network"])
    n["measurement_month"] = pd.to_datetime(n.measurement_month)
    n["report_available_at"] = pd.to_datetime(n.report_available_at, format="ISO8601").dt.tz_convert(config.TZ)
    for c in ["probe_count", "failed_probe_count"]:
        n[c] = n[c].astype(int)
    for c in ["download_mbps_p50", "download_mbps_p90"]:
        n[c] = n[c].astype(float)
    ctx = read_raw_csv(config.SOURCES["business_context"])
    for c in ["event_start", "event_end"]:
        ctx[c] = pd.to_datetime(ctx[c])
    ctx["announced_at"] = pd.to_datetime(ctx.announced_at, format="ISO8601").dt.tz_convert(config.TZ)
    return n, ctx


def network_availability(network: pd.DataFrame, dates) -> pd.DataFrame:
    """Ключ (region, measurement_month, snapshot_at): месяц измерений = месяц предыдущего биллинга.

    Значения доли ошибок заполнены, только если отчёт опубликован не позже snapshot_at.
    """
    rows = []
    for T in dates:
        T_msk = pd.Timestamp(T).tz_localize(config.TZ)
        month = pd.Timestamp(T) - pd.offsets.MonthBegin(1)
        part = network[network.measurement_month == month].copy()
        part["snapshot_at"] = pd.Timestamp(T)
        part["report_known_at_snapshot"] = part.report_available_at <= T_msk
        rate = part.failed_probe_count / part.probe_count
        part["failed_probe_rate_known"] = rate.where(part.report_known_at_snapshot)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out[["region", "measurement_month", "snapshot_at", "report_available_at", "report_known_at_snapshot",
                "probe_count", "failed_probe_count", "failed_probe_rate_known"]]


# ---------- групповые доли ----------

def churn_table(df: pd.DataFrame, by) -> pd.DataFrame:
    """n_observations, n_customers, n_churn, churn_rate по строкам «абонент × T» с известной меткой."""
    d = df[df.churn_target.notna()]
    t = d.groupby(by, observed=True).agg(n_observations=("customer_id", "size"),
                                         n_customers=("customer_id", "nunique"),
                                         n_churn=("churn_target", "sum")).reset_index()
    t["n_churn"] = t.n_churn.astype(int)
    t["churn_rate"] = (t.n_churn / t.n_observations).round(5)
    t["small_group"] = t.n_observations < config.SMALL_GROUP
    return t


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def num(x):
    return f"{x:,.0f}".replace(",", " ")


def save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def table(df, name):
    write_csv(df, config.TABLES / f"{name}.csv")


def month_label(ts):
    names = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
    return f"{names[ts.month - 1]} {str(ts.year)[2:]}"


# ---------- рисунки ----------

def fig1_missing(s):
    fields = ["tariff"] + NUMERIC + ["region", "tenure_days"]
    n = len(s)
    reasons = {"traffic_gb": "пропуск в биллинге (неизвестный объём)",
               "voice_minutes": "отрицательное значение в источнике заменено пропуском"}
    t = pd.DataFrame({"field": fields, "n_rows": n, "n_missing": [int(s[f].isna().sum()) for f in fields]})
    t["missing_share"] = (t.n_missing / n).round(5)
    t["reason"] = t.field.map(reasons).fillna("")
    table(t, "fig1_missing")
    t = t.sort_values("missing_share")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(t.field, t.missing_share * 100, color=SERIES[0], height=0.6)
    for y, (m, sh) in enumerate(zip(t.n_missing, t.missing_share)):
        ax.text(sh * 100 + 0.02, y, f"{m} ({pct(sh, 2)})", va="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, max(t.missing_share.max() * 100 * 1.6, 0.5))
    ax.set_xlabel("Доля пропусков, % строк «абонент × T»")
    ax.grid(axis="y", visible=False)
    ax.set_title("Рис. 1. Пропуски исследуемых показателей")
    fig.text(0.0, -0.06, f"Период T: 01.12.2025–01.07.2026, строк: {num(n)}. traffic_gb — пропуск в биллинге; "
             "voice_minutes — некорректное (отрицательное) значение,\nзаменённое пропуском на этапе 2. Пропуски не "
             "заменяются нулём. Счётчики поддержки — 0 при отсутствии доступных обращений, а не пропуск.",
             fontsize=8, color=INK2)
    save(fig, "fig1_missing")


def describe(series):
    k = series.dropna()
    q = k.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    return {"n_known": len(k), "n_missing": int(series.isna().sum()), "mean": k.mean(), "min": k.min(),
            "p25": q[0.25], "median": q[0.5], "p75": q[0.75], "p90": q[0.9], "p99": q[0.99], "max": k.max(),
            "share_zero": (k == 0).mean()}


def fig2_distributions(s):
    stats = pd.DataFrame({c: describe(s[c]) for c in NUMERIC + ["tenure_days"]}).T.round(3)
    stats.index.name = "field"
    pos = s.closing_debt_rub[s.closing_debt_rub > 0]
    stats.loc["closing_debt_rub (только >0)"] = pd.Series(describe(pos)).round(3)
    table(stats.reset_index(), "fig2_numeric_summary")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    tr = s.traffic_gb.dropna()
    a1.hist(tr, bins=np.arange(0, tr.max() + 5, 5), color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
    a1.axvline(tr.median(), color=INK, lw=1, ls="--")
    a1.text(tr.median() + 3, a1.get_ylim()[1] * 0.9, f"медиана {tr.median():.1f} ГБ".replace(".", ","), color=INK)
    a1.set_xlabel("Трафик за последний доступный месяц, ГБ")
    a1.set_ylabel("Строк «абонент × T»")
    a1.set_title("Рис. 2а. Трафик")
    a1.text(0.98, 0.65, f"известно: {num(len(tr))}\nпропусков: {num(s.traffic_gb.isna().sum())}\n"
            f"p90 = {tr.quantile(.9):.1f}\np99 = {tr.quantile(.99):.1f}\nmax = {tr.max():.1f}".replace(".", ","),
            transform=a1.transAxes, ha="right", va="top", color=INK2, fontsize=8.5)

    debt = s.closing_debt_rub.dropna()
    a2.hist(pos, bins=np.logspace(np.log10(pos.min()), np.log10(pos.max()), 35), color=SERIES[1], edgecolor=SURFACE,
            linewidth=0.8)
    a2.set_xscale("log")
    a2.axvline(pos.median(), color=INK, lw=1, ls="--")
    a2.set_xlabel("Долг на конец последнего доступного месяца, руб. (лог. шкала, только >0)")
    a2.set_title("Рис. 2б. Долг")
    a2.text(0.02, 0.97, f"известно: {num(len(debt))}, пропусков: {num(s.closing_debt_rub.isna().sum())}\n"
            f"долг = 0: {pct((debt == 0).mean())} строк\nдолг > 0: {num(len(pos))} строк\n"
            f"медиана (>0) = {num(pos.median())} руб.\np90 (>0) = {num(pos.quantile(.9))} руб.\n"
            f"max = {num(pos.max())} руб.", transform=a2.transAxes, va="top", color=INK2, fontsize=8.5)
    fig.text(0.0, -0.04, "Период T: 01.12.2025–01.07.2026. Значения — из строки биллинга за месяц T−1 "
             "(доступна в 00:00 T). Длинный правый хвост долга — допустимые значения, а не ошибки.\n"
             "Пики долга около 450 и 750 руб. совпадают с ценой тарифов Basic и Comfort: это неоплаченный месяц.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig2_distributions")


def fig3_dynamics(s, base_check):
    t = churn_table(s, "snapshot_at")
    t = t.merge(base_check, on="snapshot_at", how="left")
    t["target_end_exclusive"] = t.snapshot_at + pd.offsets.MonthBegin(1)
    table(t.assign(snapshot_at=fmt_date(t.snapshot_at), target_end_exclusive=fmt_date(t.target_end_exclusive)),
          "fig3_churn_by_T")
    x = np.arange(len(t))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    a1.plot(x, t.n_observations, color=SERIES[0], lw=2, marker="o", ms=6)
    for i, v in enumerate(t.n_observations):
        a1.text(i, v + 4, num(v), ha="center", color=INK2, fontsize=8.5)
    a1.set_ylim(t.n_observations.min() - 20, t.n_observations.max() + 20)
    a1.set_ylabel("Активная база, абонентов")
    a1.set_title("Рис. 3. Размер активной базы и доля оттока по датам расчёта T")
    a1.grid(axis="x", visible=False)
    a2.plot(x, t.churn_rate * 100, color=SERIES[1], lw=2, marker="o", ms=6)
    for i, (c, n, r) in enumerate(zip(t.n_churn, t.n_observations, t.churn_rate)):
        a2.text(i, r * 100 + 0.08, f"{pct(r, 2)}\n{c}/{num(n)}", ha="center", color=INK2, fontsize=8)
    a2.set_ylim(0, max(t.churn_rate * 100) + 0.8)
    a2.set_ylabel("Доля оттока за [T, T+1 мес.), %")
    a2.set_xticks(x, [f"01.{d.month:02d}.{str(d.year)[2:]}" for d in t.snapshot_at])
    a2.set_xlabel("Дата расчёта T")
    a2.grid(axis="x", visible=False)
    fig.text(0.0, -0.05, "Подписи: числитель (расторжения в горизонте) / знаменатель (активная база на T). База: "
             "activation < T и (нет termination или termination ≥ T).\nОдин абонент входит в несколько дат; "
             "строки не являются независимыми наблюдениями.", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig3_churn_dynamics")
    return t


def _bars(ax, t, col, order, colors, title, xlabel):
    t = t.set_index(col).reindex(order).dropna(subset=["n_observations"]).reset_index()
    x = np.arange(len(t))
    ax.bar(x, t.churn_rate * 100, color=colors[:len(t)], width=0.6)
    for i, r in t.iterrows():
        mark = " *" if r.small_group else ""
        ax.text(i, r.churn_rate * 100 + 0.15, f"{pct(r.churn_rate)}{mark}\n{int(r.n_churn)}/{num(r.n_observations)}",
                ha="center", color=INK2, fontsize=8.5)
    ax.set_xticks(x, t[col])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def fig4_debt_support(s):
    debt = churn_table(s, "debt_group")
    sup = churn_table(s, "support_group")
    table(debt, "fig4_churn_by_debt")
    table(sup, "fig4_churn_by_support")
    by_t = churn_table(s, ["snapshot_at", "debt_group"])
    table(by_t.assign(snapshot_at=fmt_date(by_t.snapshot_at)), "fig4_churn_by_debt_within_T")
    by_tariff = churn_table(s, ["tariff", "debt_group"])
    table(by_tariff, "fig4_churn_by_debt_within_tariff")
    net = churn_table(s.assign(network_group=np.select([s.support_network_count == 0, s.support_network_count == 1],
                                                       ["0", "1"], "2+")), "network_group")
    table(net, "fig4_churn_by_network_support")
    bill = churn_table(s.assign(billing_group=np.select([s.support_billing_count == 0, s.support_billing_count == 1],
                                                        ["0", "1"], "2+")), "billing_group")
    table(bill, "fig4_churn_by_billing_support")
    delay = churn_table(s.assign(delay_group=pd.cut(s.payment_delay_days, [-1, 0, 27, 31],
                                                    labels=["0", "1–27", "28–31 (нет оплаты)"])), "delay_group")
    table(delay, "extra_churn_by_payment_delay")
    paid0 = churn_table(s.assign(paid_zero=np.where(s.paid_rub == 0, "оплата = 0", "оплата > 0")), "paid_zero")
    table(paid0, "extra_churn_by_zero_payment")
    usage = s.groupby("churn_target")[["traffic_gb", "voice_minutes", "charged_rub", "payment_delay_days"]].median()
    table(usage.reset_index(), "extra_median_usage_by_target")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    _bars(a1, debt, "debt_group", ["=0", ">0", "unknown"], [SERIES[0], SERIES[1], MUTED],
          "Рис. 4а. Отток по долгу на T", "Долг на конец месяца T−1, руб.")
    a1.set_ylabel("Доля оттока за [T, T+1 мес.), %")
    _bars(a2, sup, "support_group", ["0", "1", "2+", "unknown"], [SERIES[0]] * 3 + [MUTED],
          "Рис. 4б. Отток по числу обращений", "Доступные обращения, созданные в месяце T−1")
    a1.set_ylim(0, debt.churn_rate.max() * 100 + 1.5)
    fig.text(0.0, -0.06, "Подписи: доля, n_churn / n_observations (строки «абонент × T», T = 01.12.2025–01.07.2026). "
             "Неизвестных значений долга и обращений\nв исследуемой базе нет — у всех активных абонентов есть строка "
             "биллинга за T−1. * — группа меньше 30 строк.", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig4_debt_support")
    return debt, sup, by_t, by_tariff


def fig5_tenure(s):
    ten = churn_table(s, "tenure_group")
    strat = churn_table(s, ["tenure_group", "debt_group"])
    tar = churn_table(s, "tariff")
    reg = churn_table(s, "region")
    hist = churn_table(s.assign(history=pd.cut(s.billing_months_available, [0, 2, 5, 12],
                                               labels=["1–2 мес.", "3–5 мес.", "6–12 мес."])), "history")
    table(ten, "fig5_churn_by_tenure")
    table(strat, "fig5_churn_by_tenure_within_debt")
    table(tar, "extra_churn_by_tariff")
    table(reg, "extra_churn_by_region")
    table(hist, "extra_churn_by_billing_history")
    debt_share = s.groupby("tenure_group").apply(lambda g: (g.closing_debt_rub > 0).mean(), include_groups=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, gridspec_kw={"width_ratios": [1, 1.6]})
    _bars(a1, ten, "tenure_group", ["<90 дней", "≥90 дней"], [SERIES[2], SERIES[0]],
          "Рис. 5а. Отток по стажу на T", "Стаж = T − activation_date")
    a1.set_ylabel("Доля оттока за [T, T+1 мес.), %")
    w = 0.36
    for j, (grp, color) in enumerate([("<90 дней", SERIES[2]), ("≥90 дней", SERIES[0])]):
        part = strat[strat.tenure_group == grp].set_index("debt_group").reindex(["=0", ">0"])
        xs = np.arange(2) + (j - 0.5) * w
        a2.bar(xs, part.churn_rate * 100, width=w - 0.03, color=color, label=grp)
        for xv, (_, r) in zip(xs, part.iterrows()):
            a2.text(xv, r.churn_rate * 100 + 0.15, f"{pct(r.churn_rate)}\n{int(r.n_churn)}/{num(r.n_observations)}",
                    ha="center", color=INK2, fontsize=8)
    a2.set_xticks(np.arange(2), ["долг = 0", "долг > 0"])
    a2.set_title("Рис. 5б. Стаж внутри групп долга")
    a2.legend(title="Стаж", loc="upper left")
    a2.grid(axis="x", visible=False)
    a1.set_ylim(0, strat.churn_rate.max() * 100 + 2)
    fig.text(0.0, -0.06, f"Доля строк с долгом > 0: <90 дней — {pct(debt_share['<90 дней'])}, ≥90 дней — "
             f"{pct(debt_share['≥90 дней'])}. Группа <90 дней мала ({int(ten.n_churn[0])} уходов): оценка неустойчива."
             "\nГраница 90 дней задана до расчёта. Строки «абонент × T» повторяют одних абонентов.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig5_tenure")
    return ten, strat, tar, reg, hist


def fig6_network(network, ctx, availability):
    months = (pd.DatetimeIndex(config.EDA_DATES) - pd.offsets.MonthBegin(1))
    n = network[network.measurement_month.isin(months)].copy()
    n["snapshot_at"] = n.measurement_month + pd.offsets.MonthBegin(1)
    n["report_known_at_snapshot"] = n.report_available_at <= n.snapshot_at.dt.tz_localize(config.TZ)
    n["failed_probe_rate"] = n.failed_probe_count / n.probe_count
    total = n.groupby("measurement_month")[["failed_probe_count", "probe_count"]].sum()
    total["failed_probe_rate"] = total.failed_probe_count / total.probe_count
    out = n.copy()
    out["publication_delay_hours"] = ((out.report_available_at - out.snapshot_at.dt.tz_localize(config.TZ))
                                      .dt.total_seconds() / 3600).round(1)
    table(out.assign(measurement_month=fmt_date(out.measurement_month), snapshot_at=fmt_date(out.snapshot_at),
                     report_available_at=fmt_ts(out.report_available_at),
                     report_known_at_snapshot=fmt_bool(out.report_known_at_snapshot),
                     failed_probe_rate=out.failed_probe_rate.round(5)), "fig6_network_by_region")
    table(total.reset_index().assign(measurement_month=lambda d: fmt_date(d.measurement_month),
                                     failed_probe_rate=lambda d: d.failed_probe_rate.round(5)), "fig6_network_total")
    reg_all = network.groupby("region")[["failed_probe_count", "probe_count"]].sum()
    reg_all["failed_probe_rate"] = (reg_all.failed_probe_count / reg_all.probe_count).round(5)
    table(reg_all.reset_index(), "fig6_network_region_total_12m")

    fig, ax = plt.subplots(figsize=(10.5, 5))
    for region in config.REGIONS:
        r = n[n.region == region].sort_values("measurement_month")
        c = REGION_COLOR[region]
        ax.plot(r.measurement_month, r.failed_probe_rate * 100, color=c, lw=2, zorder=2)
        k = r[r.report_known_at_snapshot]
        u = r[~r.report_known_at_snapshot]
        ax.scatter(k.measurement_month, k.failed_probe_rate * 100, s=60, color=c, edgecolor=c, lw=1.6, zorder=3)
        ax.scatter(u.measurement_month, u.failed_probe_rate * 100, s=42, facecolor=SURFACE, edgecolor=c, lw=1.6,
                   zorder=3)
        last = r.iloc[-1]
        ax.text(last.measurement_month + pd.Timedelta(days=6), last.failed_probe_rate * 100, region, color=INK2,
                va="center", fontsize=8.5)
    ax.plot(total.index, total.failed_probe_rate * 100, color=INK, lw=1.2, ls="--", zorder=1,
            label="все регионы: Σ неуспешных / Σ проб")
    for _, e in ctx[ctx.event_type == "measurement_alert"].iterrows():
        last_month = e.event_end.to_period("M").to_timestamp()
        ax.axvspan(e.event_start - pd.Timedelta(days=15), last_month + pd.Timedelta(days=15), color=GRID, alpha=0.6,
                   zorder=0)
        ax.text(e.event_start - pd.Timedelta(days=13), 6.75,
                f"{e.context_id}: {e.region}\nопубл. {e.announced_at:%d.%m.%Y}",
                fontsize=7.5, color=INK2, va="top")
    ax.set_xticks(months, [month_label(m) for m in months])
    ax.set_xlim(months[0] - pd.Timedelta(days=20), months[-1] + pd.Timedelta(days=75))
    ax.set_ylabel("Доля неуспешных проб, %")
    ax.set_xlabel("Месяц измерений (= месяц биллинга T−1 для T = 1-е число следующего месяца)")
    ax.set_title("Рис. 6. Доля неуспешных технических проб по регионам")
    ax.scatter([], [], s=60, color=MUTED, label="отчёт опубликован к T (можно использовать)")
    ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=MUTED, lw=1.6, label="отчёт опубликован после T (на T неизвестен)")
    ax.set_ylim(0, 7)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    known = int(availability.report_known_at_snapshot.sum())
    fig.text(0.0, -0.1, f"Пробы за месяц: {num(n.probe_count.min())}–{num(n.probe_count.max())} на регион. "
             f"На датах EDA известны {known} из {len(availability)} отчётов (регион × T). Ретроспективно показаны "
             "все отчёты;\nдля сравнения с оттоком допустимы только закрашенные точки. Серые полосы — периоды "
             "предупреждений из business_context (контекст, не причина ухода).", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig6_network")
    return n, total


def network_vs_churn(s, availability):
    """Ретроспективное сопоставление с оттоком только по известным на T отчётам (регион × T)."""
    j = s.merge(availability[["region", "snapshot_at", "report_known_at_snapshot", "failed_probe_rate_known"]],
                on=["region", "snapshot_at"], how="left", validate="many_to_one")
    j["network_report"] = np.where(j.report_known_at_snapshot, "известен на T", "неизвестен на T")
    t = churn_table(j, "network_report")
    known = j[j.report_known_at_snapshot]
    med = availability.failed_probe_rate_known.median()
    k = churn_table(known.assign(fail_level=np.where(known.failed_probe_rate_known > med, "выше медианы",
                                                     "не выше медианы")), "fail_level")
    table(t, "fig6_churn_by_report_availability")
    table(k.assign(threshold=round(med, 5)), "fig6_churn_by_known_fail_level")
    return t, k


# ---------- проверки ----------

def independent_base_counts(dates) -> pd.DataFrame:
    """Активная база, пересчитанная напрямую из сырого журнала событий (без кода snapshots)."""
    e = read_raw_csv(config.SOURCES["events"]).drop_duplicates()
    act = pd.to_datetime(e[e.event_type == "activation"].set_index("customer_id").event_date)
    ter = pd.to_datetime(e[e.event_type == "termination"].set_index("customer_id").event_date).reindex(act.index)
    return pd.DataFrame({"snapshot_at": list(dates),
                         "independent_base": [int(((act < T) & (ter.isna() | (ter >= T))).sum()) for T in dates]})


def boundary_example() -> pd.DataFrame:
    """Искусственный пример событий ровно на T и ровно на конце горизонта."""
    T = pd.Timestamp("2026-03-01")
    end = snapshots.horizon_end(T)
    contract = pd.DataFrame({
        "activation_date": [T, T - pd.Timedelta(days=1), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"),
                            pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        "termination_date": [pd.NaT, pd.NaT, T, end, end - pd.Timedelta(days=1), T - pd.Timedelta(days=1)],
    }, index=["A_act_on_T", "B_act_before_T", "C_term_on_T", "D_term_on_end", "E_term_before_end", "F_term_before_T"])
    expected = {"A_act_on_T": (False, None), "B_act_before_T": (True, 0), "C_term_on_T": (True, 1),
                "D_term_on_end": (True, 0), "E_term_before_end": (True, 1), "F_term_before_T": (False, None)}
    base = set(snapshots.active_base(contract, T))
    sample = pd.DataFrame({"customer_id": sorted(base), "snapshot_at": T})
    lab = snapshots.add_target(sample, contract).set_index("customer_id").churn_target
    rows = []
    for cid, (in_base, label) in expected.items():
        got_label = lab.get(cid, None)
        got_label = None if got_label is None or pd.isna(got_label) else int(got_label)
        rows.append({"case": cid, "expected_in_base": in_base, "got_in_base": cid in base, "expected_label": label,
                     "got_label": got_label, "ok": (cid in base) == in_base and got_label == label})
    return pd.DataFrame(rows)


def run_checks(s, features, base_check, availability, groups) -> pd.DataFrame:
    res = []

    def add(name, ok, detail):
        res.append({"check": name, "result": "OK" if ok else "FAIL", "detail": detail})

    add("Ключ (customer_id, snapshot_at) уникален", not s.duplicated(["customer_id", "snapshot_at"]).any(),
        f"{len(s)} строк")
    m = s.groupby("snapshot_at").size().rename("n").reset_index().merge(base_check, on="snapshot_at")
    add("Число строк по T = независимо рассчитанная активная база", bool((m.n == m.independent_base).all()),
        "; ".join(f"{d:%d.%m.%Y}: {a}={b}" for d, a, b in zip(m.snapshot_at, m.n, m.independent_base)))
    T_msk = s.snapshot_at.dt.tz_localize(config.TZ)
    add("billing_available_at ≤ snapshot_at", bool((s.billing_available_at <= T_msk).all()),
        f"нарушений: {int((s.billing_available_at > T_msk).sum())}")
    add("source_billing_month = месяц T−1", bool((s.source_billing_month == s.snapshot_at - pd.offsets.MonthBegin(1))
                                                 .all()), "стандартный комплект")
    leak = [c for c in features.columns if "termination" in c or c in ("churn_target", "target_end_exclusive")]
    add("Дата будущего расторжения и метка не входят в признаки", not leak, f"столбцы признаков: {len(features.columns)}")
    add("Все метки EDA наблюдаемы (конец горизонта ≤ 01.10.2026)", bool(s.label_observed.all()),
        f"неизвестных меток: {int((~s.label_observed).sum())}")
    b = boundary_example()
    add("Граничный пример: события ровно на T и на конце горизонта", bool(b.ok.all()),
        "; ".join(f"{r.case}: база={r.got_in_base}, метка={r.got_label}" for r in b.itertuples()))
    late = availability[availability.report_available_at > availability.snapshot_at.dt.tz_localize(config.TZ)]
    add("Сетевой отчёт, опубликованный после T, неизвестен на T",
        bool((~late.report_known_at_snapshot).all() and late.failed_probe_rate_known.isna().all()),
        f"поздних отчётов: {len(late)} из {len(availability)}")
    total_n, total_c = len(s), int(s.churn_target.sum())
    for name, g in groups.items():
        add(f"Числители и знаменатели восстанавливаются: {name}",
            int(g.n_observations.sum()) == total_n and int(g.n_churn.sum()) == total_c,
            f"Σn={int(g.n_observations.sum())} из {total_n}, Σchurn={int(g.n_churn.sum())} из {total_c}")
    return pd.DataFrame(res), b


# ---------- запуск ----------

DICTIONARY = [
    ("customer_id", "Идентификатор абонента", "—", "ключ"),
    ("snapshot_at", "Дата расчёта T, 00:00 Москвы", "дата", "ключ"),
    ("split", "Часть разбиения по времени", "—", "eda (в этой таблице только EDA-даты)"),
    ("source_billing_month", "Месяц биллинга, из которого взяты показатели", "месяц", "последний доступный к T (T−1)"),
    ("billing_available_at", "Момент доступности этого месяца", "дата и время", "≤ snapshot_at"),
    ("billing_months_available", "Число месяцев биллинга, доступных к T", "мес.", "длина истории"),
    ("tariff ... tariff_changes_count", "Показатели биллинга за source_billing_month", "см. словарь этапа 2", "на T"),
    ("traffic_gb_missing, voice_minutes_invalid", "Флаги пропуска/некорректности", "0/1", "на T"),
    ("support_*_count", "Доступные обращения, созданные в source_billing_month", "шт.", "available_at ≤ T"),
    ("region, acquisition_channel, customer_segment, activation_date", "Постоянные характеристики", "—", "постоянные"),
    ("tenure_days, tenure_group", "Стаж на T и группа (<90 / ≥90 дней)", "дни", "на T"),
    ("debt_group", "Группа долга по closing_debt_rub: >0, =0, unknown", "—", "на T"),
    ("support_group", "Группа support_count: 0, 1, 2+, unknown", "—", "на T"),
    ("target_end_exclusive", "Конец горизонта метки (исключён)", "дата", "метка, не признак"),
    ("label_observed", "Конец горизонта ≤ 01.10.2026", "0/1", "метка, не признак"),
    ("churn_target", "Расторжение в [snapshot_at, target_end_exclusive)", "0/1", "метка, не признак"),
]


def serialize_sample(s):
    out = s.copy()
    for c in ["snapshot_at", "source_billing_month", "activation_date", "target_end_exclusive"]:
        out[c] = fmt_date(out[c])
    out["billing_available_at"] = fmt_ts(out.billing_available_at)
    for c in ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub", "traffic_gb"]:
        out[c] = fmt_money(out[c])
    for c in FLAGS + ["label_observed"]:
        out[c] = fmt_bool(out[c].astype(bool))
    for c in ["voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count", "support_count",
              "support_network_count", "support_billing_count", "support_unknown_topic_count", "churn_target"]:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else str(int(x)))
    order = ["customer_id", "snapshot_at", "split", "source_billing_month", "billing_available_at",
             "billing_months_available"] + snapshots.FEATURES + snapshots.STATIC + [
        "tenure_days", "tenure_group", "debt_group", "support_group", "target_end_exclusive", "label_observed",
        "churn_target"]
    return out[order]


def run():
    events, subs, cm = load_prepared()
    network, ctx = load_network()
    contract = snapshots.contract_table(events)

    # Шаг 1: признаки на T (без метки). Шаг 2: метка отдельно.
    features = snapshots.build_features(contract, subs, cm, config.EDA_DATES)
    s = snapshots.add_target(features, contract)
    s = s[s.label_observed].reset_index(drop=True)  # на EDA-датах неизвестных меток нет; правило сохранено явно

    availability = network_availability(network, config.EDA_DATES)
    base_check = independent_base_counts(config.EDA_DATES)

    write_csv(serialize_sample(s), config.EDA_DATA / "eda_sample.csv")
    write_csv(pd.DataFrame(DICTIONARY, columns=["field", "description", "unit", "availability"]),
              config.EDA_DATA / "eda_sample_dictionary.csv")
    write_csv(availability.assign(measurement_month=fmt_date(availability.measurement_month),
                                  snapshot_at=fmt_date(availability.snapshot_at),
                                  report_available_at=fmt_ts(availability.report_available_at),
                                  report_known_at_snapshot=fmt_bool(availability.report_known_at_snapshot),
                                  failed_probe_rate_known=availability.failed_probe_rate_known.round(5)),
              config.EDA_DATA / "network_availability.csv")

    cats = pd.concat([s[c].value_counts(dropna=False).rename_axis("value").rename("n").reset_index().assign(field=c)
                      for c in ["tariff", "region", "acquisition_channel", "customer_segment", "tenure_group",
                                "debt_group", "support_group"]])
    table(cats[["field", "value", "n"]], "category_frequencies")

    fig1_missing(s)
    fig2_distributions(s)
    dyn = fig3_dynamics(s, base_check)
    debt, sup, debt_t, debt_tariff = fig4_debt_support(s)
    ten, strat, tar, reg, hist = fig5_tenure(s)
    fig6_network(network, ctx, availability)
    network_vs_churn(s, availability)
    ctx_out = ctx.assign(event_start=fmt_date(ctx.event_start), event_end=fmt_date(ctx.event_end),
                         announced_at=fmt_ts(ctx.announced_at),
                         announced_after_period_end=fmt_bool(ctx.announced_at.dt.tz_localize(None) >
                                                             ctx.event_end + pd.Timedelta(days=1)))
    table(ctx_out, "business_context_timeline")

    checks, boundary = run_checks(s, features, base_check, availability,
                                  {"T": dyn, "долг": debt, "обращения": sup, "стаж": ten, "тариф": tar,
                                   "регион": reg, "стаж×долг": strat})
    write_csv(checks, config.REPORTS_EDA / "checks.csv")
    write_csv(boundary, config.REPORTS_EDA / "boundary_example.csv")

    run_config = {
        "eda_dates": [f"{d:%Y-%m-%d}" for d in config.EDA_DATES],
        "validation_dates": [f"{d:%Y-%m-%d}" for d in config.VALIDATION_DATES],
        "test_dates": [f"{d:%Y-%m-%d}" for d in config.TEST_DATES],
        "horizon": config.HORIZON, "active_base_rule": config.ACTIVE_BASE_RULE,
        "events_observed_to": f"{config.EVENTS_OBSERVED_TO:%Y-%m-%d}",
        "rows": len(s), "customers": int(s.customer_id.nunique()), "churn": int(s.churn_target.sum()),
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "eda_sample_sha256": hashlib.sha256((config.EDA_DATA / "eda_sample.csv").read_bytes()).hexdigest(),
    }
    (config.REPORTS_EDA / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    return run_config, checks


def main():
    run_config, checks = run()
    print(json.dumps({k: run_config[k] for k in ["rows", "customers", "churn"]}, ensure_ascii=False))
    print(checks[["result", "check"]].to_string(index=False))
    if (checks.result != "OK").any():
        raise SystemExit("есть непройденные проверки")


if __name__ == "__main__":
    main()
