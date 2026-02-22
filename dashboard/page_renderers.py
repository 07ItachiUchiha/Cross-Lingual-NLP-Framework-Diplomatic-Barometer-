from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


def _build_dashboard_decision_hygiene(_df: pd.DataFrame, _scored_df: pd.DataFrame, _yearly_df: pd.DataFrame, _report: Dict):
    warnings = []
    notes = []

    if _df is None or len(_df) == 0:
        return {"warnings": ["No documents loaded."], "notes": [], "evidence": pd.DataFrame()}

    year_min = int(_yearly_df["year"].min()) if _yearly_df is not None and len(_yearly_df) else None
    year_max = int(_yearly_df["year"].max()) if _yearly_df is not None and len(_yearly_df) else None

    if year_min is not None and year_max is not None and "year" in _df.columns:
        expected_years = set(range(year_min, year_max + 1))
        present_years = set(int(x) for x in _df["year"].dropna().unique())
        missing_years = sorted(list(expected_years - present_years))
    else:
        missing_years = []

    docs_per_year = _df.groupby("year").size().sort_index().to_dict() if "year" in _df.columns else {}
    low_volume_years = [int(y) for y, c in docs_per_year.items() if int(c) < 3]

    source_counts = _df["source"].value_counts(dropna=False).to_dict() if "source" in _df.columns else {}
    total_docs = int(len(_df))
    dominant_source = None
    dominant_ratio = 0.0
    if total_docs > 0 and source_counts:
        for source, count in source_counts.items():
            ratio = float(count) / float(total_docs)
            if ratio > dominant_ratio:
                dominant_ratio = ratio
                dominant_source = str(source)

    if total_docs < 30:
        warnings.append(f"Small corpus: only {total_docs} documents.")
    if missing_years:
        warnings.append(f"Coverage gap (missing years): {missing_years[:20]}" + ("…" if len(missing_years) > 20 else ""))
    if low_volume_years:
        warnings.append(f"Low-volume years (<3 docs): {sorted(set(low_volume_years))[:20]}" + ("…" if len(set(low_volume_years)) > 20 else ""))
    if dominant_source is not None and dominant_ratio >= 0.80:
        warnings.append(f"Source imbalance: {dominant_source} is {dominant_ratio:.0%} of documents.")

    crossover_year = _report.get("crossover_year")
    if crossover_year is None:
        notes.append("No crossover year detected; shift may be gradual or data sparse.")

    evidence_df = pd.DataFrame()
    if _scored_df is not None and len(_scored_df) > 0:
        working = _scored_df.copy()
        if "date" in working.columns:
            working["date"] = pd.to_datetime(working["date"], errors="coerce")
        if "year" not in working.columns and "date" in working.columns:
            working["year"] = working["date"].dt.year

        working["economic_score"] = pd.to_numeric(working.get("economic_score"), errors="coerce").fillna(0.0)
        working["security_score"] = pd.to_numeric(working.get("security_score"), errors="coerce").fillna(0.0)
        working["shift_signal"] = working["security_score"] - working["economic_score"]

        if crossover_year is not None and "year" in working.columns:
            focus = working[working["year"].isin([int(crossover_year) - 1, int(crossover_year), int(crossover_year) + 1])]
            if len(focus) >= 5:
                working = focus

        if "url" not in working.columns or working["url"].astype(str).str.strip().eq("").all():
            try:
                project_root = Path(__file__).resolve().parent.parent
                raw_dir = project_root / "data" / "raw"
                candidates = sorted(raw_dir.glob("live_scrape_*.csv"), reverse=True)
                if candidates:
                    live_df = pd.read_csv(candidates[0])
                    if {"date", "title", "source", "url"}.issubset(set(live_df.columns)):
                        live_df = live_df.copy()
                        live_df["_key_date"] = pd.to_datetime(live_df["date"], errors="coerce").dt.date.astype(str)
                        live_df["_key_title"] = live_df["title"].astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
                        live_df["_key_source"] = live_df["source"].astype(str).str.upper().str.strip()
                        live_df["url"] = live_df["url"].astype(str).str.strip()
                        live_df = live_df[live_df["url"].ne("")]
                        lookup = {
                            (r["_key_date"], r["_key_title"], r["_key_source"]): r["url"]
                            for _, r in live_df.iterrows()
                        }

                        if "url" not in working.columns:
                            working["url"] = ""
                        working["_key_date"] = working["date"].dt.date.astype(str)
                        working["_key_title"] = working["title"].astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
                        working["_key_source"] = working["source"].astype(str).str.upper().str.strip()

                        def fill_url(row):
                            cur = str(row.get("url", "")).strip()
                            if cur:
                                return cur
                            return lookup.get((row.get("_key_date", ""), row.get("_key_title", ""), row.get("_key_source", "")), "")

                        working["url"] = working.apply(fill_url, axis=1)
                        working.drop(columns=["_key_date", "_key_title", "_key_source"], inplace=True, errors="ignore")
            except Exception:
                pass

        cols = [c for c in ["date", "year", "title", "source", "url", "economic_score", "security_score", "shift_signal"] if c in working.columns]
        evidence_df = working.sort_values("shift_signal", ascending=False).head(10)[cols]

    return {"warnings": warnings, "notes": notes, "evidence": evidence_df}


def _compute_chart_adequacy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    min_years: int = 4,
    min_points: int = 8,
    min_variance: float = 1e-4,
) -> Dict[str, object]:
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or x_col not in df.columns or y_col not in df.columns:
        return {"years": 0, "points": 0, "missing_pct": 100.0, "variance": 0.0, "low_info": True, "reasons": ["insufficient data"]}

    working = df.copy()
    missing_pct = float(working[[x_col, y_col]].isna().any(axis=1).mean() * 100.0)
    working = working.dropna(subset=[x_col, y_col])
    years = int(pd.to_numeric(working[x_col], errors="coerce").dropna().nunique())
    points = int(len(working))
    variance = float(pd.to_numeric(working[y_col], errors="coerce").dropna().var()) if points > 1 else 0.0

    reasons = []
    if years < min_years:
        reasons.append(f"only {years} years")
    if points < min_points:
        reasons.append(f"only {points} points")
    if variance < min_variance:
        reasons.append("near-flat signal")

    return {
        "years": years,
        "points": points,
        "missing_pct": round(missing_pct, 1),
        "variance": variance,
        "low_info": len(reasons) > 0,
        "reasons": reasons,
    }


def render_overview_page(ctx: Dict[str, Any]) -> None:
    st = ctx["st"]
    lang = ctx["lang"]
    country1_name = ctx["country1_name"]
    country2_name = ctx["country2_name"]
    run_meta = ctx["run_meta"]
    exploratory_mode = ctx["exploratory_mode"]
    processed_df = ctx["processed_df"]
    role_profile = ctx["role_profile"]
    year_min = ctx["year_min"]
    year_max = ctx["year_max"]
    external_ctx = ctx["external_ctx"]
    baseline_processed_df = ctx["baseline_processed_df"]

    render_page_scaffold = ctx["render_page_scaffold"]
    get_analysis_bundle = ctx["get_analysis_bundle"]
    render_top_takeaways = ctx["render_top_takeaways"]
    summarize_issue_counts = ctx["summarize_issue_counts"]
    summarize_equity_dimensions = ctx["summarize_equity_dimensions"]
    render_external_status_metrics = ctx["render_external_status_metrics"]
    build_live_geopolitical_pulse = ctx["build_live_geopolitical_pulse"]
    render_data_basis_caption = ctx["render_data_basis_caption"]
    build_claim_traceability = ctx["build_claim_traceability"]
    triggers_to_dataframe = ctx["triggers_to_dataframe"]
    build_overview_verdicts = ctx["build_overview_verdicts"]
    perform_quality_audit = ctx["perform_quality_audit"]
    build_pilot_pack = ctx["build_pilot_pack"]
    render_page_exports = ctx["render_page_exports"]

    st.header(f"{country1_name}-{country2_name} Diplomatic Overview")
    render_page_scaffold(
        question=f"What are the highest-confidence diplomatic decisions for {country1_name}-{country2_name} this month?",
        run_meta=run_meta,
        exploratory_mode=exploratory_mode,
    )

    st.markdown(f"""
    How do {country1_name} and {country2_name} communications vary over time across **economic** and
    **security** themes? This dashboard summarizes patterns in {len(processed_df)}
    official diplomatic documents from {year_min} to {year_max}.
    """)

    if role_profile == "Think Tank":
        st.info("Priority: focus on confidence, traceability, and significance before claiming directional change.")
    elif role_profile == "NGO":
        st.info("Priority: focus on equity impact lens, issue distribution, and vulnerable/community-sensitive signals.")
    elif role_profile == "Diplomat":
        st.info("Priority: focus on contradiction scan, source profiles, and year-to-year narrative shifts.")
    else:
        st.info("Priority: focus on policy triggers, decision option cards, and confidence-weighted interpretation.")

    overview_bundle = get_analysis_bundle(include_tone_theme=True)
    overview_report = overview_bundle.get("report", {})
    overview_scored_df = overview_bundle.get("scored_df", pd.DataFrame())
    overview_yearly_df = overview_bundle.get("yearly_df", pd.DataFrame())
    overview_stats_result = overview_bundle.get("stats_result", {})
    overview_confidence = overview_bundle.get("confidence_summary", {})
    overview_contradiction_df = overview_bundle.get("contradiction_df", pd.DataFrame())
    overview_source_credibility_score = float(overview_bundle.get("corpus_credibility_score", 0.0))
    overview_weighted_focus = overview_bundle.get("weighted_focus", {})
    overview_trigger_rows = overview_bundle.get("trigger_rows", [])
    overview_decision_options_df = overview_bundle.get("decision_options_df", pd.DataFrame())

    st.markdown("#### Action")
    st.subheader("Top 3 decisions this month")
    if isinstance(overview_decision_options_df, pd.DataFrame) and len(overview_decision_options_df) > 0:
        top3 = overview_decision_options_df.head(3)
        for _, row in top3.iterrows():
            st.info(
                f"**{row.get('option', 'Option')}** — {row.get('recommended_action', 'No action provided')} "
                f"(Risk: {row.get('risk_level', 'Unknown')}; Evidence: {row.get('evidence_anchor', 'n/a')})"
            )
    else:
        st.info("No decision options available for current filter window.")

    render_top_takeaways(
        role_profile=role_profile,
        page_key="overview",
        context={
            "docs": len(processed_df),
            "years": processed_df["year"].nunique() if "year" in processed_df.columns else 0,
            "confidence_label": overview_confidence.get("label", "Unknown"),
            "trend": overview_report.get("trend", "Unknown"),
            "top_issue": "n/a",
        },
    )

    if role_profile == "NGO":
        st.markdown("#### NGO Impact Summary")
        ngo_issue_counts = summarize_issue_counts(processed_df)
        ngo_region_eq, ngo_group_eq = summarize_equity_dimensions(processed_df)
        top_issue = (
            str(ngo_issue_counts.sort_values("documents", ascending=False).iloc[0]["issue"])
            if isinstance(ngo_issue_counts, pd.DataFrame) and len(ngo_issue_counts) > 0 and "issue" in ngo_issue_counts.columns
            else "No dominant issue"
        )
        top_region = (
            str(ngo_region_eq.sort_values("documents", ascending=False).iloc[0]["equity_dimension"])
            if isinstance(ngo_region_eq, pd.DataFrame) and len(ngo_region_eq) > 0 and "equity_dimension" in ngo_region_eq.columns
            else "No region signal"
        )
        top_group = (
            str(ngo_group_eq.sort_values("documents", ascending=False).iloc[0]["equity_dimension"])
            if isinstance(ngo_group_eq, pd.DataFrame) and len(ngo_group_eq) > 0 and "equity_dimension" in ngo_group_eq.columns
            else "No community signal"
        )
        st.info(
            f"Who/Where affected: region focus = {top_region}; community focus = {top_group}. "
            f"Trend direction: {overview_report.get('trend', 'Unknown')}. "
            f"Dominant issue area: {top_issue}."
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(lang["total_documents"], len(processed_df))
    with col2:
        total_words = processed_df['cleaned'].apply(lambda x: len(str(x).split())).sum()
        st.metric(lang["total_words"], f"{total_words:,}")
    with col3:
        most_active_year = processed_df['year'].mode()[0]
        st.metric("Most Active Year", most_active_year)
    with col4:
        year_count = len(processed_df['year'].unique())
        st.metric("Years Covered", year_count)

    st.subheader("External Integration Status")
    render_external_status_metrics(external_ctx)

    with st.expander("External source details", expanded=False):
        detail_rows = []
        for key, label in [
            ("comtrade_yearly", "Comtrade"),
            ("estat_rows", "e-Stat"),
            ("ogd_rows", "OGD India"),
            ("rss_year", "RSS"),
            ("external_signals", "External Signals"),
        ]:
            item = external_ctx.get(key, {})
            detail_rows.append({
                "source": label,
                "rows": item.get("rows", 0),
                "latest_file": item.get("file_name", ""),
                "status": "Loaded" if item.get("available") else "Not available",
            })
        st.dataframe(pd.DataFrame(detail_rows), width='stretch', hide_index=True)

        rss_report_info = external_ctx.get("rss_report", {})
        rss_payload = rss_report_info.get("data", {}) if isinstance(rss_report_info, dict) else {}
        diagnostics = rss_payload.get("feed_diagnostics", []) if isinstance(rss_payload, dict) else []
        if diagnostics:
            diag_rows = []
            for item in diagnostics:
                diag_rows.append(
                    {
                        "feed_url": item.get("feed_url", ""),
                        "http_status": item.get("http_status", ""),
                        "entries_seen": item.get("entries_seen", 0),
                        "error": str(item.get("error", ""))[:180],
                        "final_url": item.get("final_url", ""),
                    }
                )
            st.caption("RSS feed diagnostics (latest run)")
            st.dataframe(pd.DataFrame(diag_rows), width='stretch', hide_index=True)

    st.subheader("Live Geopolitical Pulse")
    pulse = build_live_geopolitical_pulse(external_ctx)
    if pulse.get("available"):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Events (24h)", int(pulse.get("count_24h", 0)))
        with p2:
            st.metric("Events (7d)", int(pulse.get("count_7d", 0)))
        with p3:
            st.metric("Providers (7d)", int(pulse.get("providers", 0)))
        with p4:
            st.metric("Pulse Confidence", f"{pulse.get('confidence_label', 'Low')} ({pulse.get('confidence_score', 0):.1f}/100)", delta=pulse.get("momentum_label", "Stable"))

        vc1, vc2 = st.columns(2)
        with vc1:
            daily_counts = pulse.get("daily_counts_7d", [])
            if daily_counts:
                daily_df = pd.DataFrame(daily_counts)
                fig_daily = px.line(daily_df, x="date", y="count", markers=True, title="7-Day Activity Trend")
                st.plotly_chart(fig_daily, width='stretch')
                render_data_basis_caption(run_meta, "Live pulse trend", point_count=len(daily_df))

        with vc2:
            recency_split = pulse.get("recency_split", {})
            if isinstance(recency_split, dict) and recency_split:
                recency_df = pd.DataFrame(
                    [
                        {"window": "Last 24h", "count": int(recency_split.get("last_24h", 0))},
                        {"window": "Prior 6d", "count": int(recency_split.get("prior_6d", 0))},
                    ]
                )
                st.caption("Recency mix (7d)")
                st.dataframe(recency_df, width='stretch', hide_index=True)

        provider_counts = pulse.get("provider_counts", {})
        if isinstance(provider_counts, dict) and provider_counts:
            providers_df = pd.DataFrame(list(provider_counts.items()), columns=["provider", "count"]).sort_values("count", ascending=False)
            st.caption("Provider distribution (7d)")
            st.dataframe(providers_df, width='stretch', hide_index=True)

        top_events = pulse.get("top_events", [])
        if top_events:
            with st.expander("Top events (latest 7d)", expanded=False):
                st.dataframe(pd.DataFrame(top_events), width='stretch', hide_index=True)
    else:
        st.info("Live geopolitical pulse unavailable (external signals not loaded).")

    st.markdown("---")

    if "doc_type" in processed_df.columns:
        with st.expander("Document types", expanded=False):
            counts = (
                processed_df["doc_type"].fillna("Unknown").astype(str).value_counts().reset_index()
            )
            counts.columns = ["doc_type", "documents"]
            st.dataframe(counts, width='stretch', hide_index=True)

    st.subheader("The Strategic Shift: Economic vs. Security Focus Over Time")

    report, scored_df, yearly_df = overview_report, overview_scored_df, overview_yearly_df
    stats_result = overview_stats_result
    confidence_summary = overview_confidence
    traceability_df = build_claim_traceability(scored_df, yearly_df)
    source_cred_df = overview_bundle.get("source_cred_df", pd.DataFrame())
    corpus_credibility_score = overview_source_credibility_score
    weighted_focus = overview_weighted_focus
    contradiction_df = overview_contradiction_df
    tone_df = overview_bundle.get("tone_df", pd.DataFrame())
    thematic_analysis = overview_bundle.get("thematic_analysis", {})

    trigger_df = triggers_to_dataframe(overview_trigger_rows)
    decision_options_df = overview_decision_options_df

    hygiene = _build_dashboard_decision_hygiene(processed_df, scored_df, yearly_df, report)
    if hygiene.get("warnings"):
        st.subheader("Decision hygiene (for policy use)")
        for w in hygiene["warnings"]:
            st.warning(w)
        for n in hygiene.get("notes", []):
            st.info(n)
        ev = hygiene.get("evidence")
        if isinstance(ev, pd.DataFrame) and len(ev) > 0:
            st.markdown("**Evidence (top documents behind the shift signal)**")
            st.dataframe(ev, width='stretch')

            export_df = ev.copy()
            if "review_decision" not in export_df.columns:
                export_df["review_decision"] = ""
            if "review_notes" not in export_df.columns:
                export_df["review_notes"] = ""

            csv_bytes = export_df.to_csv(index=False, encoding="utf-8").encode("utf-8")
            st.download_button(
                label="Download evidence (CSV for review)",
                data=csv_bytes,
                file_name=f"evidence_review_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_df['year'],
        y=yearly_df['economic_score_mean'],
        mode='lines+markers',
        name='Economic Focus',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Economic: %{y:.4f}<br>Docs: %{customdata}<extra></extra>',
        customdata=yearly_df.get('economic_score_count', None)
    ))

    if 'economic_score_std' in yearly_df.columns:
        econ_mean = yearly_df['economic_score_mean']
        econ_std = yearly_df['economic_score_std']
        fig.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=(econ_mean + econ_std),
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=(econ_mean - econ_std),
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(46,134,171,0.12)',
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=yearly_df['year'],
        y=yearly_df['security_score_mean'],
        mode='lines+markers',
        name='Security Focus',
        line=dict(color='#A23B72', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Security: %{y:.4f}<br>Docs: %{customdata}<extra></extra>',
        customdata=yearly_df.get('security_score_count', None)
    ))

    if 'security_score_std' in yearly_df.columns:
        sec_mean = yearly_df['security_score_mean']
        sec_std = yearly_df['security_score_std']
        fig.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=(sec_mean + sec_std),
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=(sec_mean - sec_std),
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(162,59,114,0.12)',
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="Diplomatic Focus Signals Over Time (Lexicon-based)",
        xaxis_title="Year",
        yaxis_title="Normalized Focus Score",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        font=dict(size=12)
    )

    st.plotly_chart(fig, width='stretch')
    render_data_basis_caption(run_meta, "Strategic shift trend", point_count=len(yearly_df))

    if isinstance(report, dict) and report.get('by_source'):
        with st.expander("View per-source summary (MEA vs MOFA)"):
            st.caption("These are computed from the same real documents, split by the 'source' column.")
            rows = []
            for src, r in report.get('by_source', {}).items():
                rows.append({
                    'source': src,
                    'documents': r.get('total_documents'),
                    'economic_avg': r.get('overall_economic_avg'),
                    'security_avg': r.get('overall_security_avg'),
                    'trend': r.get('trend'),
                    'crossover_year': r.get('crossover_year'),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), width='stretch')

    st.markdown(f"### {lang['key_findings']}")
    col1, col2 = st.columns(2)

    economic_avg = float(report.get('overall_economic_avg', 0.0))
    security_avg = float(report.get('overall_security_avg', 0.0))
    gap = security_avg - economic_avg
    if gap > 0:
        dominant_focus = f"Security-led (+{gap:.4f})"
    elif gap < 0:
        dominant_focus = f"Economic-led (+{abs(gap):.4f})"
    else:
        dominant_focus = "Balanced"

    with col1:
        st.markdown(f"""
        <div class="stat-box">
        <b>{lang['overall_economic']}:</b> {economic_avg:.4f}<br/>
        <b>{lang['overall_security']}:</b> {security_avg:.4f}<br/>
        <b>Credibility-weighted Economic:</b> {weighted_focus.get('weighted_economic_mean', 0.0):.4f}<br/>
        <b>Credibility-weighted Security:</b> {weighted_focus.get('weighted_security_mean', 0.0):.4f}<br/>
        <b>Dominant Focus:</b> {dominant_focus}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if report['crossover_year']:
            st.markdown(f"""
            <div class="stat-box">
            <b>{lang['crossover_year']}:</b> {report['crossover_year']}<br/>
            {lang['security_overtook']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(lang['strategic_shift_evident'])

        st.write(f"**{lang['overall_trend']}:** {report['trend']}")
        if stats_result:
            st.caption(
                f"Stats check ({stats_result.get('preferred_test')}): "
                f"p={stats_result.get('preferred_p_value', float('nan')):.4g}, "
                f"effect={stats_result.get('effect_size', 0.0):.3f}"
            )
        st.caption(f"Corpus source credibility score: {corpus_credibility_score:.3f} (0 to 1)")

    st.markdown("### Claim Confidence")
    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        st.metric("Evidence Confidence", confidence_summary.get("label", "Low"))
        st.metric("Confidence Score", f"{confidence_summary.get('score', 0):.2f}/100")
    with c2:
        comp = confidence_summary.get("components", {})
        comp_df = pd.DataFrame([
            {"component": "Sample size", "score": comp.get("sample_size", 0.0)},
            {"component": "Source diversity", "score": comp.get("source_diversity", 0.0)},
            {"component": "Source credibility", "score": comp.get("source_credibility", 0.0)},
            {"component": "Temporal coverage", "score": comp.get("temporal_coverage", 0.0)},
            {"component": "Statistical strength", "score": comp.get("statistical_strength", 0.0)},
            {"component": "External coverage", "score": comp.get("external_coverage", 0.0)},
        ])
        st.dataframe(comp_df, width='stretch', hide_index=True)

    with st.expander("Source credibility profile", expanded=False):
        if isinstance(source_cred_df, pd.DataFrame) and len(source_cred_df) > 0:
            src_show = source_cred_df.copy()
            src_show["credibility_score"] = pd.to_numeric(src_show["credibility_score"], errors="coerce").round(3)
            st.dataframe(src_show, width='stretch', hide_index=True)
            fig_src_cred = px.bar(
                src_show,
                x="source",
                y="credibility_score",
                title="Credibility score by source",
            )
            st.plotly_chart(fig_src_cred, width='stretch')
        else:
            st.info("No source credibility profile available.")

    with st.expander("Claim Traceability (year-by-year)", expanded=False):
        if isinstance(traceability_df, pd.DataFrame) and len(traceability_df) > 0:
            st.dataframe(traceability_df, width='stretch', hide_index=True)
            st.download_button(
                label="Download traceability (CSV)",
                data=traceability_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"claim_traceability_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No traceability records available for this dataset.")

    with st.expander("Potential cross-source contradictions", expanded=False):
        if isinstance(contradiction_df, pd.DataFrame) and len(contradiction_df) > 0:
            st.dataframe(contradiction_df.head(50), width='stretch', hide_index=True)
            st.download_button(
                label="Download contradictions (CSV)",
                data=contradiction_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"contradictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No potential contradictions detected in current corpus.")

    st.markdown("### Policy Trigger Alerts")
    if isinstance(trigger_df, pd.DataFrame) and len(trigger_df) > 0:
        for _, row in trigger_df.iterrows():
            sev = str(row.get("severity", "Info")).lower()
            msg = f"{row.get('trigger', '')}: {row.get('condition', '')}"
            action = f"Action: {row.get('policy_action', '')}"
            if sev == "high":
                st.error(f"{msg} — {action}")
            elif sev == "medium":
                st.warning(f"{msg} — {action}")
            else:
                st.info(f"{msg} — {action}")

        with st.expander("Policy trigger table", expanded=False):
            st.dataframe(trigger_df, width='stretch', hide_index=True)
            st.download_button(
                label="Download policy triggers (CSV)",
                data=trigger_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"policy_triggers_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    st.markdown("### Unified Verdict (all major sections)")
    verdict_rows = build_overview_verdicts(
        processed_df=processed_df,
        report=report,
        scored_df=scored_df,
        yearly_df=yearly_df,
        stats_result=stats_result,
        tone_df=tone_df,
        thematic_analysis=thematic_analysis,
        external_ctx=external_ctx,
        contradiction_df=contradiction_df,
    )
    st.dataframe(pd.DataFrame(verdict_rows), width='stretch', hide_index=True)

    st.markdown("### Decision Option Cards")
    if isinstance(decision_options_df, pd.DataFrame) and len(decision_options_df) > 0:
        for _, opt in decision_options_df.iterrows():
            risk = str(opt.get("risk_level", "Info"))
            card_text = (
                f"**{opt.get('option', '')}**\n\n"
                f"When to use: {opt.get('when_to_use', '')}\n\n"
                f"Action: {opt.get('recommended_action', '')}\n\n"
                f"Evidence: {opt.get('evidence_anchor', '')}"
            )
            if risk.lower() == "high":
                st.error(card_text)
            elif risk.lower() == "medium":
                st.warning(card_text)
            else:
                st.info(card_text)

        with st.expander("Decision options table", expanded=False):
            st.dataframe(decision_options_df, width='stretch', hide_index=True)
            st.download_button(
                label="Download decision options (CSV)",
                data=decision_options_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"decision_options_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    st.markdown("### Quarterly Quality Audit")
    q_summary_df = overview_bundle.get("q_summary_df", pd.DataFrame())
    q_issue_df = overview_bundle.get("q_issue_df", pd.DataFrame())
    q_meta = overview_bundle.get("q_meta", {})
    _, _, q_meta_full = perform_quality_audit(baseline_processed_df)

    qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
    with qa_col1:
        st.metric("Audit Score", f"{q_meta.get('audit_score', 0):.1f}/100")
    with qa_col2:
        st.metric("Audit Status", str(q_meta.get("status", "Unknown")))
    with qa_col3:
        st.metric("Quarters", int(q_meta.get("quarters_analyzed", 0)))
    with qa_col4:
        st.metric("Issues", int(q_meta.get("issues_count", 0)))

    st.caption(
        "Audit metrics above use the current filtered dataset "
        f"({len(processed_df)} docs). Full-corpus reference: "
        f"{q_meta_full.get('audit_score', 0):.1f}/100, "
        f"{q_meta_full.get('status', 'Unknown')} "
        f"across {q_meta_full.get('quarters_analyzed', 0)} quarters."
    )

    if isinstance(q_issue_df, pd.DataFrame) and len(q_issue_df) > 0:
        high_count = int((q_issue_df["severity"].astype(str).str.lower() == "high").sum())
        med_count = int((q_issue_df["severity"].astype(str).str.lower() == "medium").sum())
        low_count = int((q_issue_df["severity"].astype(str).str.lower() == "low").sum())
        if high_count > 0:
            st.error(f"Quality audit alerts: {high_count} high, {med_count} medium, {low_count} low.")
        elif med_count > 0:
            st.warning(f"Quality audit alerts: {high_count} high, {med_count} medium, {low_count} low.")
        else:
            st.info(f"Quality audit alerts: {high_count} high, {med_count} medium, {low_count} low.")

        with st.expander("Quarterly quality issues", expanded=False):
            st.dataframe(q_issue_df, width='stretch', hide_index=True)
    else:
        st.info("No quarterly quality issues detected.")

    with st.expander("Quarterly quality summary", expanded=False):
        if isinstance(q_summary_df, pd.DataFrame) and len(q_summary_df) > 0:
            st.dataframe(q_summary_df, width='stretch', hide_index=True)
            q_fig_docs = px.bar(q_summary_df, x="quarter", y="documents", title="Documents by quarter")
            st.plotly_chart(q_fig_docs, width='stretch')
            q_fig_url = px.line(q_summary_df, x="quarter", y="missing_url_pct", markers=True, title="Missing URL % by quarter")
            st.plotly_chart(q_fig_url, width='stretch')

            st.download_button(
                label="Download quarterly audit summary (CSV)",
                data=q_summary_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"quarterly_quality_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
            if isinstance(q_issue_df, pd.DataFrame) and len(q_issue_df) > 0:
                st.download_button(
                    label="Download quarterly audit issues (CSV)",
                    data=q_issue_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                    file_name=f"quarterly_quality_issues_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("Quarterly summary unavailable for current filtered data.")

    st.markdown("### Pilot Validation Readiness")
    pilot_checklist_df, pilot_log_template_df, pilot_meta = build_pilot_pack(
        processed_df=processed_df,
        trigger_df=trigger_df,
        decision_options_df=decision_options_df,
        confidence_summary=confidence_summary,
        quality_meta=q_meta,
    )

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.metric("Pilot Readiness", pilot_meta.get("readiness_label", "Needs Prep"))
    with pcol2:
        st.metric("Readiness Score", f"{pilot_meta.get('readiness_score', 0):.1f}/100")
    with pcol3:
        st.metric("Pilot Tasks", len(pilot_checklist_df) if isinstance(pilot_checklist_df, pd.DataFrame) else 0)

    with st.expander("Pilot task checklist", expanded=False):
        if isinstance(pilot_checklist_df, pd.DataFrame) and len(pilot_checklist_df) > 0:
            st.dataframe(pilot_checklist_df, width='stretch', hide_index=True)
            st.download_button(
                label="Download pilot checklist (CSV)",
                data=pilot_checklist_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"pilot_checklist_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with st.expander("Pilot session log template", expanded=False):
        if isinstance(pilot_log_template_df, pd.DataFrame) and len(pilot_log_template_df) > 0:
            st.dataframe(pilot_log_template_df, width='stretch', hide_index=True)
            st.download_button(
                label="Download pilot session log template (CSV)",
                data=pilot_log_template_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"pilot_session_log_template_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    st.markdown("### Issue Landscape")
    issue_counts_view = summarize_issue_counts(processed_df)
    if isinstance(issue_counts_view, pd.DataFrame) and len(issue_counts_view) > 0:
        col_issue_1, col_issue_2 = st.columns([0.45, 0.55])
        with col_issue_1:
            st.dataframe(issue_counts_view, width='stretch', hide_index=True)
        with col_issue_2:
            fig_issue = px.bar(
                issue_counts_view.head(8),
                x="issue",
                y="documents",
                title="Top issue tags in corpus",
            )
            st.plotly_chart(fig_issue, width='stretch')

        issue_trends_view = ctx["summarize_issue_trends"](processed_df)
        if isinstance(issue_trends_view, pd.DataFrame) and len(issue_trends_view) > 0:
            trend_df = issue_trends_view.copy()
            trend_df["year"] = pd.to_numeric(trend_df.get("year"), errors="coerce")
            trend_df["documents"] = pd.to_numeric(trend_df.get("documents"), errors="coerce")
            trend_df = trend_df.dropna(subset=["year", "documents", "issue"])

            if len(trend_df) > 0:
                top_n = 6
                top_issues = (
                    trend_df.groupby("issue", as_index=False)["documents"]
                    .sum()
                    .sort_values("documents", ascending=False)
                    .head(top_n)["issue"]
                    .astype(str)
                    .tolist()
                )
                trend_df["issue_grouped"] = trend_df["issue"].astype(str).where(trend_df["issue"].astype(str).isin(top_issues), "Other")
                grouped = (
                    trend_df.groupby(["year", "issue_grouped"], as_index=False)["documents"]
                    .sum()
                    .sort_values(["year", "documents"], ascending=[True, False])
                )

                adequacy = _compute_chart_adequacy(grouped, "year", "documents", min_years=4, min_points=10, min_variance=1e-3)
                st.caption("Data adequacy — Issue trends")
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.metric("Years", int(adequacy.get("years", 0)))
                with a2:
                    st.metric("Points", int(adequacy.get("points", 0)))
                with a3:
                    st.metric("Missing %", f"{float(adequacy.get('missing_pct', 0.0)):.1f}")
                with a4:
                    st.metric("Variance", f"{float(adequacy.get('variance', 0.0)):.4f}")

                if adequacy.get("low_info", False):
                    reasons = ", ".join(adequacy.get("reasons", []))
                    st.info(f"Fallback: Issue trend chart hidden due to low-information signal ({reasons}).")
                    fallback = grouped.groupby("issue_grouped", as_index=False)["documents"].sum().sort_values("documents", ascending=False)
                    st.dataframe(fallback, width='stretch', hide_index=True)
                else:
                    fig_issue_trend = px.line(
                        grouped,
                        x="year",
                        y="documents",
                        color="issue_grouped",
                        title="Issue tag trends over time (Top issues + Other)",
                        markers=True,
                    )
                    st.plotly_chart(fig_issue_trend, width='stretch')
            else:
                st.info("Fallback: Issue trend visual unavailable because trend rows are empty after cleaning.")
    else:
        st.info("No issue tags detected for the current filtered dataset.")

    st.markdown("### Equity Impact Lens")
    region_equity_view, group_equity_view = summarize_equity_dimensions(processed_df)
    eq_col1, eq_col2 = st.columns(2)

    with eq_col1:
        st.markdown("**Regional impact dimensions**")
        if isinstance(region_equity_view, pd.DataFrame) and len(region_equity_view) > 0:
            st.dataframe(region_equity_view, width='stretch', hide_index=True)
        else:
            st.info("No region-sensitive equity signals detected.")

    with eq_col2:
        st.markdown("**Community impact dimensions**")
        if isinstance(group_equity_view, pd.DataFrame) and len(group_equity_view) > 0:
            st.dataframe(group_equity_view, width='stretch', hide_index=True)
        else:
            st.info("No community-sensitive equity signals detected.")

    if isinstance(region_equity_view, pd.DataFrame) and len(region_equity_view) > 0:
        fig_eq_r = px.bar(
            region_equity_view,
            x="equity_dimension",
            y="documents",
            title="Equity regional dimensions",
        )
        st.plotly_chart(fig_eq_r, width='stretch')

    if isinstance(group_equity_view, pd.DataFrame) and len(group_equity_view) > 0:
        fig_eq_g = px.bar(
            group_equity_view,
            x="equity_dimension",
            y="documents",
            title="Equity community dimensions",
        )
        st.plotly_chart(fig_eq_g, width='stretch')

    st.info("""
    **How to read this chart:**

    - **Blue line (Economic):** Higher = more talk about trade, investment, infrastructure
    - **Pink line (Security):** Higher = more talk about defense, military, strategic alliances
    - **If the lines cross:** This suggests a change in emphasis within the available document set
    - **What it indicates:** a lexicon-based signal (verify with evidence rows and coverage warnings)
    """)

    render_page_exports(
        page_slug="overview",
        tables={
            "overview_verdicts": pd.DataFrame(verdict_rows),
            "yearly_shift": yearly_df if isinstance(yearly_df, pd.DataFrame) else pd.DataFrame(),
            "policy_triggers": trigger_df if isinstance(trigger_df, pd.DataFrame) else pd.DataFrame(),
            "decision_options": decision_options_df if isinstance(decision_options_df, pd.DataFrame) else pd.DataFrame(),
            "quality_summary": q_summary_df if isinstance(q_summary_df, pd.DataFrame) else pd.DataFrame(),
        },
        run_meta=run_meta,
    )


def render_stats_page(ctx: Dict[str, Any]) -> None:
    st = ctx["st"]
    pd_local = pd
    stats_local = stats

    country1_name = ctx["country1_name"]
    country2_name = ctx["country2_name"]
    run_meta = ctx["run_meta"]
    exploratory_mode = ctx["exploratory_mode"]
    processed_df = ctx["processed_df"]
    role_profile = ctx["role_profile"]
    external_ctx = ctx["external_ctx"]

    render_page_scaffold = ctx["render_page_scaffold"]
    get_analysis_bundle = ctx["get_analysis_bundle"]
    render_top_takeaways = ctx["render_top_takeaways"]
    build_claim_traceability = ctx["build_claim_traceability"]
    render_page_exports = ctx["render_page_exports"]
    MIN_DOCS_SIGNIFICANCE = ctx["MIN_DOCS_SIGNIFICANCE"]

    st.header(f"Statistical Checks")
    render_page_scaffold(
        question=f"Can current evidence be trusted for policy action, or should it remain exploratory?",
        run_meta=run_meta,
        exploratory_mode=exploratory_mode,
    )
    st.markdown(f"Are differences in signals likely to be meaningful, or could they be noise? These checks summarize the evidence.")

    if role_profile == "Think Tank":
        st.caption("Think Tank view: prioritize preferred test p-value, effect size, and coverage quality before publication.")
    elif role_profile == "NGO":
        st.caption("NGO view: combine statistical checks with equity/community filter outcomes before impact framing.")
    elif role_profile == "Diplomat":
        st.caption("Diplomat view: combine significance signals with contradiction and source-profile context for negotiation posture.")
    else:
        st.caption("Policy view: use trigger rules + decision options as action layer on top of statistical diagnostics.")

    stats_bundle = get_analysis_bundle(include_tone_theme=False)
    report = stats_bundle.get("report", {})
    scored_df = stats_bundle.get("scored_df", pd_local.DataFrame())
    yearly_df = stats_bundle.get("yearly_df", pd_local.DataFrame())
    stats_result_preview = stats_bundle.get("stats_result", {})
    render_top_takeaways(
        role_profile=role_profile,
        page_key="stats",
        context={
            "docs": len(processed_df),
            "years": processed_df["year"].nunique() if "year" in processed_df.columns else 0,
            "confidence_label": "n/a",
            "p_value": (f"{stats_result_preview.get('preferred_p_value', float('nan')):.4g}" if stats_result_preview else None),
            "effect": (f"{stats_result_preview.get('effect_size', 0.0):.3f}" if stats_result_preview else None),
            "trend": report.get("trend", "Unknown"),
            "top_issue": "n/a",
        },
    )

    try:
        year_min = int(pd_local.to_datetime(scored_df["date"], errors="coerce").dt.year.min())
        year_max = int(pd_local.to_datetime(scored_df["date"], errors="coerce").dt.year.max())
    except Exception:
        year_min, year_max = None, None

    st.subheader("Structured-source context")
    comtrade_info = external_ctx.get("comtrade_yearly", {})
    if comtrade_info.get("available"):
        cdf = comtrade_info.get("data", pd_local.DataFrame()).copy()
        if "refYear" in cdf.columns:
            cdf["refYear"] = pd_local.to_numeric(cdf["refYear"], errors="coerce")
            comtrade_years = sorted([int(y) for y in cdf["refYear"].dropna().unique()])
            if year_min is not None and year_max is not None:
                corpus_years = set(range(int(year_min), int(year_max) + 1))
                overlap = sorted(list(corpus_years.intersection(set(comtrade_years))))
                st.caption(f"Comtrade years overlap with corpus: {overlap if overlap else 'None'}")
        st.caption(f"Comtrade source: {comtrade_info.get('file_name', '')}")
    else:
        st.caption("Comtrade context unavailable.")

    st.subheader("Economic vs. Security Focus Comparison")

    stats_result = stats_bundle.get("stats_result", {})
    if len(scored_df) < MIN_DOCS_SIGNIFICANCE:
        st.warning(
            f"Significance checks use {len(scored_df)} docs; recommended minimum is {MIN_DOCS_SIGNIFICANCE} for stable inference.",
            icon="⚠️",
        )
    confidence_summary = stats_bundle.get("confidence_summary", {})
    traceability_df = build_claim_traceability(scored_df, yearly_df)
    source_cred_df = stats_bundle.get("source_cred_df", pd_local.DataFrame())
    corpus_credibility_score = float(stats_bundle.get("corpus_credibility_score", 0.0))
    weighted_focus = stats_bundle.get("weighted_focus", {})
    trigger_df = stats_bundle.get("trigger_df", pd_local.DataFrame())
    decision_options_df = stats_bundle.get("decision_options_df", pd_local.DataFrame())

    st.subheader("Evidence confidence")
    conf_col1, conf_col2 = st.columns([0.35, 0.65])
    with conf_col1:
        st.metric("Confidence Label", confidence_summary.get("label", "Low"))
        st.metric("Confidence Score", f"{confidence_summary.get('score', 0):.2f}/100")
    with conf_col2:
        comp = confidence_summary.get("components", {})
        st.caption(
            f"Sample={comp.get('sample_size', 0):.1f}, Sources={comp.get('source_diversity', 0):.1f}, "
            f"Credibility={comp.get('source_credibility', 0):.1f}, "
            f"Time={comp.get('temporal_coverage', 0):.1f}, Stats={comp.get('statistical_strength', 0):.1f}, "
            f"External={comp.get('external_coverage', 0):.1f}"
        )
        st.caption(
            f"Corpus source credibility (0-1): {corpus_credibility_score:.3f}; "
            f"Weighted gap (security-economic): {weighted_focus.get('weighted_gap_security_minus_economic', 0.0):.4f}"
        )

    with st.expander("Source credibility profile", expanded=False):
        if isinstance(source_cred_df, pd_local.DataFrame) and len(source_cred_df) > 0:
            src = source_cred_df.copy()
            src["credibility_score"] = pd_local.to_numeric(src["credibility_score"], errors="coerce").round(3)
            st.dataframe(src, width='stretch', hide_index=True)
        else:
            st.info("No source credibility records available.")

    if stats_result:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Paired N", f"{stats_result.get('paired_n', 0)}")
        with col2:
            st.metric("Test Used", f"{stats_result.get('preferred_test', 'paired_t')}")
        with col3:
            st.metric("P-Value", f"{stats_result.get('preferred_p_value', float('nan')):.6f}")
        with col4:
            st.metric("Significant", "Yes" if stats_result['significant'] else "No")
        with col5:
            st.metric("Effect Size", f"{stats_result['effect_size']:.4f}")

        st.caption(
            f"Mean difference (security - economic): {stats_result.get('mean_difference_security_minus_economic', 0.0):.4f}; "
            f"95% CI [{stats_result.get('ci95_low', 0.0):.4f}, {stats_result.get('ci95_high', 0.0):.4f}]"
        )
        if stats_result.get("normality_p") is not None:
            st.caption(f"Normality check p-value on paired differences: {stats_result['normality_p']:.4f}")
        if stats_result.get("wilcoxon_p") is not None:
            st.caption(f"Wilcoxon signed-rank p-value: {stats_result['wilcoxon_p']:.6f}")

        if stats_result['significant']:
            if exploratory_mode:
                st.warning("Difference appears statistically significant, but current coverage is exploratory; avoid strong policy claims.")
            else:
                st.success("Statistically significant paired difference detected (preferred test p < 0.05)")
        else:
            st.info("No statistically significant paired difference at the 0.05 level")

    with st.expander("Statistical claim traceability", expanded=False):
        if isinstance(traceability_df, pd_local.DataFrame) and len(traceability_df) > 0:
            st.dataframe(traceability_df, width='stretch', hide_index=True)
        else:
            st.info("No traceability records available.")

    st.subheader("Policy trigger rules")
    if isinstance(trigger_df, pd_local.DataFrame) and len(trigger_df) > 0:
        st.dataframe(trigger_df, width='stretch', hide_index=True)
    else:
        st.info("No active policy triggers.")

    st.subheader("Decision option cards")
    if isinstance(decision_options_df, pd_local.DataFrame) and len(decision_options_df) > 0:
        st.dataframe(decision_options_df, width='stretch', hide_index=True)
    else:
        st.info("No decision options available.")

    st.subheader("Quality audit snapshot")
    q_summary_df = stats_bundle.get("q_summary_df", pd_local.DataFrame())
    q_issue_df = stats_bundle.get("q_issue_df", pd_local.DataFrame())
    q_meta = stats_bundle.get("q_meta", {})
    st.caption(
        f"Audit score: {q_meta.get('audit_score', 0):.1f}/100 | "
        f"Status: {q_meta.get('status', 'Unknown')} | "
        f"Quarters: {q_meta.get('quarters_analyzed', 0)} | "
        f"Issues: {q_meta.get('issues_count', 0)}"
    )
    st.caption(
        f"Audit confidence: {q_meta.get('confidence_band', 'Unknown')} "
        f"(avg docs/quarter={q_meta.get('avg_docs_per_quarter', 0):.2f}, median={q_meta.get('median_docs_per_quarter', 0):.1f})"
    )
    if isinstance(q_issue_df, pd_local.DataFrame) and len(q_issue_df) > 0:
        st.dataframe(q_issue_df.head(20), width='stretch', hide_index=True)
    else:
        st.info("No audit issues for current view.")

    st.subheader("Year-over-Year Correlation")
    yearly_df_sorted = yearly_df.sort_values('year')
    correlation = yearly_df_sorted['economic_score_mean'].corr(yearly_df_sorted['security_score_mean'])
    st.metric("Correlation (Economic vs Security)", f"{correlation:.4f}")

    if len(yearly_df_sorted) < 5:
        st.warning("Correlation is based on a small number of years; interpret cautiously.")

    if correlation < -0.3:
        st.info("Strong negative correlation (these signals tend to move in opposite directions)")
    elif correlation > 0.3:
        st.info("Strong positive correlation (these signals tend to move together)")
    else:
        st.info("Weak correlation (no strong relationship detected)")

    st.subheader("Trend Analysis (Linear Regression)")
    X = yearly_df_sorted[['year']].values
    y_econ = yearly_df_sorted['economic_score_mean'].values
    y_sec = yearly_df_sorted['security_score_mean'].values

    year_span = float(X[-1][0] - X[0][0]) if len(X) >= 2 else 0.0
    if year_span <= 0 or len(y_econ) < 2 or len(y_sec) < 2:
        econ_slope = 0.0
        sec_slope = 0.0
        st.warning("Trend estimation needs at least two distinct years; trend values shown as 0.0.")
        econ_p = None
        sec_p = None
    else:
        econ_fit = stats_local.linregress(yearly_df_sorted['year'].astype(float), yearly_df_sorted['economic_score_mean'].astype(float))
        sec_fit = stats_local.linregress(yearly_df_sorted['year'].astype(float), yearly_df_sorted['security_score_mean'].astype(float))
        econ_slope = float(econ_fit.slope)
        sec_slope = float(sec_fit.slope)
        econ_p = float(econ_fit.pvalue)
        sec_p = float(sec_fit.pvalue)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Economic Focus Trend (per year)", f"{econ_slope:.6f}")
        if econ_p is not None:
            st.caption(f"Regression p-value: {econ_p:.4g}")
        if econ_slope < 0:
            st.write("Declining economic focus over time")
        else:
            st.write("Increasing economic focus over time")

    with col2:
        st.metric("Security Focus Trend (per year)", f"{sec_slope:.6f}")
        if sec_p is not None:
            st.caption(f"Regression p-value: {sec_p:.4g}")
        if sec_slope > 0:
            st.write("Rising security focus over time")
        else:
            st.write("Declining security focus over time")

    st.subheader("Policy action trust verdict")
    trust_label = "Proceed with policy action"
    trust_reason = []
    conf_score = float(confidence_summary.get('score', 0.0))
    audit_score = float(q_meta.get('audit_score', 0.0))
    sig_ok = bool(stats_result and stats_result.get('significant'))
    if exploratory_mode or conf_score < 55 or audit_score < 60 or not sig_ok:
        trust_label = "Use as exploratory evidence only"
    if exploratory_mode:
        trust_reason.append("coverage below robust threshold")
    if conf_score < 55:
        trust_reason.append("low confidence score")
    if audit_score < 60:
        trust_reason.append("quality audit below 60")
    if not sig_ok:
        trust_reason.append("no significant paired difference")
    st.info(f"Can we trust this shift for policy action? **{trust_label}**.")
    if trust_reason:
        st.caption("Reason(s): " + ", ".join(trust_reason))

    span_text = ""
    if year_min is not None and year_max is not None and year_max >= year_min:
        span_text = f"{year_min}–{year_max}"

    st.info(f"""
    **How to read this page:**

    - **P-value < 0.05:** This suggests the difference is unlikely under the null hypothesis (not a guarantee)
    - **Effect size:** Approximate magnitude of the difference (larger = bigger shift)
    - **Correlation:** Whether economic and security signals move together or opposite **in this dataset**
    - **Trend lines:** Long-term direction over the available span {span_text if span_text else "(dataset span varies)"}
    - **Bottom line:** Use these checks alongside coverage warnings and evidence tables
    """)

    render_page_exports(
        page_slug="statistical_checks",
        tables={
            "yearly_stats": yearly_df_sorted,
            "trigger_rules": trigger_df if isinstance(trigger_df, pd_local.DataFrame) else pd_local.DataFrame(),
            "decision_options": decision_options_df if isinstance(decision_options_df, pd_local.DataFrame) else pd_local.DataFrame(),
            "quality_issues": q_issue_df if isinstance(q_issue_df, pd_local.DataFrame) else pd_local.DataFrame(),
        },
        run_meta=run_meta,
    )
