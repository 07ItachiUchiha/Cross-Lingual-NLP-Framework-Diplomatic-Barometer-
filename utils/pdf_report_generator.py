"""
PDF report generation using ReportLab
"""

import json
from datetime import datetime
from typing import Dict, Optional, List
import logging
import os
import tempfile
from collections import Counter
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    from reportlab.lib.pagesizes import letter, A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not installed. PDF export limited. Install with: pip install reportlab")

try:
    import plotly.graph_objects as go
    from plotly.io import write_image
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly write_image not available. Install kaleido: pip install kaleido")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. PDF charts disabled.")


class PDFReportGenerator:
    """Generate comprehensive PDF reports from analysis data"""
    
    def __init__(self, title: str = "Cross-Lingual NLP Framework", subtitle: str = "India-Japan Strategic Shift Analysis"):
        self.title = title
        self.subtitle = subtitle
        self.author = "Cross-Lingual NLP Framework Team"
        self.generation_date = datetime.now()

    def _save_plot_image(self, fig, prefix: str = "pdf_chart") -> Optional[str]:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=f"{prefix}_")
            tmp_path = tmp.name
            tmp.close()
            fig.savefig(tmp_path, dpi=160, bbox_inches="tight")
            return tmp_path
        except Exception as exc:
            logger.warning(f"Failed to save chart image: {exc}")
            return None
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass

    def _classify_issue_group(self, issue_label: str) -> str:
        name = str(issue_label or "").strip().lower()
        if not name:
            return "governance"

        security_terms = ["security", "defense", "military", "maritime", "cyber", "terror", "border", "nuclear", "strategic"]
        economy_terms = ["econom", "trade", "investment", "infrastructure", "finance", "industry", "supply", "energy", "technology"]
        social_terms = ["climate", "health", "migration", "education", "culture", "human", "aid", "rights", "community", "gender"]

        if any(term in name for term in security_terms):
            return "security"
        if any(term in name for term in economy_terms):
            return "economy"
        if any(term in name for term in social_terms):
            return "social"
        return "governance"

    def _normalize_analysis_payload(self, analysis_data: Dict) -> Dict:
        if not isinstance(analysis_data, dict):
            return {}

        normalized = dict(analysis_data)

        if "tone_analysis" not in normalized and isinstance(normalized.get("tone_and_sentiment"), dict):
            tas = normalized.get("tone_and_sentiment", {})
            total_docs = normalized.get("metadata", {}).get("total_documents") if isinstance(normalized.get("metadata"), dict) else None
            normalized["tone_analysis"] = {
                "total_documents": total_docs,
                "tone_distribution": tas.get("tone_distribution", {}),
                "sentiment_distribution": tas.get("sentiment_distribution", {}),
            }

        visual_data = normalized.get("visual_data", {}) if isinstance(normalized.get("visual_data"), dict) else {}

        if not visual_data:
            visual_data = {}

            yearly_candidates = []
            strategic = normalized.get("strategic_shift", {}) if isinstance(normalized.get("strategic_shift"), dict) else {}
            for key in ["yearly_shift", "yearly_data", "aggregate_by_year", "yearly_breakdown"]:
                value = strategic.get(key)
                if isinstance(value, list) and value:
                    yearly_candidates = value
                    break
            if yearly_candidates:
                visual_data["yearly_shift"] = yearly_candidates

            tone_dist = {}
            if isinstance(normalized.get("tone_analysis"), dict):
                td = normalized["tone_analysis"].get("tone_distribution", {})
                if isinstance(td, dict):
                    tone_dist = td
            if tone_dist:
                visual_data["tone_distribution"] = tone_dist

            sentiment_dist = {}
            if isinstance(normalized.get("tone_analysis"), dict):
                sd = normalized["tone_analysis"].get("sentiment_distribution", {})
                if isinstance(sd, dict):
                    sentiment_dist = sd
            if sentiment_dist:
                visual_data["sentiment_distribution"] = sentiment_dist

            confidence_components = {}
            conf = normalized.get("confidence_summary", {}) if isinstance(normalized.get("confidence_summary"), dict) else {}
            comp = conf.get("components", {}) if isinstance(conf, dict) else {}
            if isinstance(comp, dict) and comp:
                confidence_components = comp
            if confidence_components:
                visual_data["confidence_components"] = confidence_components

            issue_tag_counts = normalized.get("issue_tag_counts", {})
            if isinstance(issue_tag_counts, dict) and issue_tag_counts:
                visual_data["issue_tag_counts"] = [
                    {"issue": str(k), "documents": float(v)} for k, v in issue_tag_counts.items()
                ]

            issue_tag_trends = normalized.get("issue_tag_trends", [])
            if isinstance(issue_tag_trends, list) and issue_tag_trends:
                visual_data["issue_tag_trends"] = issue_tag_trends

            live_pulse = normalized.get("live_pulse", {}) if isinstance(normalized.get("live_pulse"), dict) else {}
            if isinstance(live_pulse.get("daily_counts_7d"), list) and live_pulse.get("daily_counts_7d"):
                visual_data["pulse_daily_counts"] = live_pulse.get("daily_counts_7d", [])
            if isinstance(live_pulse.get("recency_split"), dict) and live_pulse.get("recency_split"):
                visual_data["pulse_recency_split"] = live_pulse.get("recency_split", {})

            theme_term_frequency = {}
            themes_obj = normalized.get("themes", {})
            if isinstance(themes_obj, dict):
                overall = themes_obj.get("overall_themes", {})
                if isinstance(overall, dict) and overall:
                    term_counter: Counter = Counter()
                    for _, terms in overall.items():
                        if isinstance(terms, list):
                            for term in terms:
                                term_str = str(term).strip()
                                if term_str:
                                    term_counter[term_str] += 1
                    if term_counter:
                        theme_term_frequency = dict(term_counter.most_common(10))
            if theme_term_frequency:
                visual_data["theme_term_frequency"] = theme_term_frequency

            if visual_data:
                normalized["visual_data"] = visual_data

        return normalized

    def _build_chart_images(self, analysis_data: Dict) -> List[str]:
        if not MATPLOTLIB_AVAILABLE:
            return []

        images: List[str] = []
        try:
            visual_data = analysis_data.get("visual_data", {}) if isinstance(analysis_data, dict) else {}
            chart_by_key: Dict[str, str] = {}

            def register_chart(key: str, path: Optional[str]) -> None:
                if path:
                    chart_by_key[key] = path

            yearly_rows = visual_data.get("yearly_shift", []) if isinstance(visual_data, dict) else []
            if yearly_rows:
                years = []
                econ = []
                sec = []
                for r in yearly_rows:
                    try:
                        y = int(r.get("year"))
                        e = float(r.get("economic_score_mean"))
                        s = float(r.get("security_score_mean"))
                    except Exception:
                        continue
                    years.append(y)
                    econ.append(e)
                    sec.append(s)

                if years and econ and sec:
                    fig, ax = plt.subplots(figsize=(8.2, 3.4))
                    ax.plot(years, econ, marker="o", label="Economic")
                    ax.plot(years, sec, marker="o", label="Security")
                    ax.set_title("Economic vs Security Focus Over Time")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Average Score")
                    ax.grid(alpha=0.25)
                    ax.legend()
                    img = self._save_plot_image(fig, "strategic_shift")
                    register_chart("01_strategic_trend", img)

            confidence_components = visual_data.get("confidence_components", {}) if isinstance(visual_data, dict) else {}
            if isinstance(confidence_components, dict) and len(confidence_components) > 0:
                labels = []
                values = []
                for k, v in confidence_components.items():
                    try:
                        labels.append(str(k).replace("_", " ").title())
                        values.append(float(v))
                    except Exception:
                        continue
                if labels and values:
                    fig, ax = plt.subplots(figsize=(8.2, 3.2))
                    ax.barh(labels, values)
                    ax.set_title("Evidence Confidence Components")
                    ax.set_xlabel("Score")
                    ax.set_xlim(0, max(100, max(values) * 1.1))
                    ax.invert_yaxis()
                    ax.grid(axis="x", alpha=0.25)
                    img = self._save_plot_image(fig, "confidence_components")
                    register_chart("02_confidence", img)

            q_rows = visual_data.get("quality_quarterly", []) if isinstance(visual_data, dict) else []
            if q_rows:
                quarters = []
                docs = []
                for r in q_rows:
                    try:
                        q = str(r.get("quarter"))
                        d = float(r.get("documents"))
                    except Exception:
                        continue
                    quarters.append(q)
                    docs.append(d)

                if quarters and docs:
                    fig, ax = plt.subplots(figsize=(8.2, 3.4))
                    ax.plot(quarters, docs, marker="o")
                    ax.set_title("Quarterly Document Coverage")
                    ax.set_xlabel("Quarter")
                    ax.set_ylabel("Documents")
                    ax.tick_params(axis="x", rotation=45)
                    ax.grid(alpha=0.25)
                    img = self._save_plot_image(fig, "quality_coverage")
                    register_chart("03_quality", img)

            tone_dist = visual_data.get("tone_distribution", {}) if isinstance(visual_data, dict) else {}
            issue_tag_counts = visual_data.get("issue_tag_counts", []) if isinstance(visual_data, dict) else []
            if isinstance(issue_tag_counts, list) and issue_tag_counts:
                labels = []
                values = []
                for row in issue_tag_counts:
                    try:
                        labels.append(str(row.get("issue", "")))
                        values.append(float(row.get("documents", 0) or 0))
                    except Exception:
                        continue
                if labels and values:
                    labels = labels[:10]
                    values = values[:10]
                    fig, ax = plt.subplots(figsize=(8.2, 3.2))
                    ax.barh(labels, values)
                    ax.set_title("Issue Tag Landscape")
                    ax.set_xlabel("Documents")
                    ax.invert_yaxis()
                    ax.grid(axis="x", alpha=0.25)
                    img = self._save_plot_image(fig, "issue_tag_landscape")
                    register_chart("04_issue_tags", img)

            issue_tag_trends = visual_data.get("issue_tag_trends", []) if isinstance(visual_data, dict) else []
            if isinstance(issue_tag_trends, list) and issue_tag_trends:
                tmp_trends = []
                for row in issue_tag_trends:
                    try:
                        tmp_trends.append(
                            {
                                "year": int(float(row.get("year"))),
                                "issue": str(row.get("issue", "")),
                                "documents": float(row.get("documents", 0) or 0),
                            }
                        )
                    except Exception:
                        continue
                if tmp_trends:
                    trend_df = pd.DataFrame(tmp_trends)
                    if len(trend_df) > 0:
                        pivot = trend_df.pivot_table(index="year", columns="issue", values="documents", aggfunc="sum", fill_value=0.0).sort_index()
                        if len(pivot) > 0:
                            top_issues = pivot.sum(axis=0).sort_values(ascending=False).head(4).index.tolist()
                            fig, ax = plt.subplots(figsize=(8.2, 3.2))
                            category_colors = {
                                "security": "#A23B72",
                                "economy": "#2E86AB",
                                "social": "#2CA58D",
                                "governance": "#6C757D",
                            }
                            style_cycle = ["-", "--", "-.", ":"]
                            for issue in top_issues:
                                category = self._classify_issue_group(str(issue))
                                color = category_colors.get(category, "#6C757D")
                                style = style_cycle[top_issues.index(issue) % len(style_cycle)]
                                ax.plot(
                                    pivot.index.astype(int),
                                    pivot[issue].values,
                                    marker="o",
                                    linestyle=style,
                                    color=color,
                                    label=f"{issue} ({category})",
                                )
                            ax.set_title("Issue Tag Trends by Year")
                            ax.set_xlabel("Year")
                            ax.set_ylabel("Documents")
                            ax.grid(alpha=0.25)
                            if top_issues:
                                ax.legend(loc="best", fontsize=8)
                            img = self._save_plot_image(fig, "issue_tag_trends")
                            register_chart("05_issue_trends", img)

            if isinstance(tone_dist, dict) and len(tone_dist) > 0:
                labels = []
                values = []
                for k, v in tone_dist.items():
                    try:
                        labels.append(str(k))
                        values.append(float(v))
                    except Exception:
                        continue
                if labels and values:
                    fig, ax = plt.subplots(figsize=(8.2, 3.2))
                    ax.bar(labels, values)
                    ax.set_title("Tone Distribution")
                    ax.set_ylabel("Documents")
                    ax.grid(axis="y", alpha=0.25)
                    img = self._save_plot_image(fig, "tone_distribution")
                    register_chart("06_tone", img)

            sentiment_dist = visual_data.get("sentiment_distribution", {}) if isinstance(visual_data, dict) else {}
            if isinstance(sentiment_dist, dict) and len(sentiment_dist) > 0:
                labels = []
                values = []
                for k, v in sentiment_dist.items():
                    try:
                        labels.append(str(k))
                        values.append(float(v))
                    except Exception:
                        continue
                if labels and values:
                    fig, ax = plt.subplots(figsize=(8.2, 3.2))
                    ax.bar(labels, values)
                    ax.set_title("Sentiment Distribution")
                    ax.set_ylabel("Documents")
                    ax.grid(axis="y", alpha=0.25)
                    img = self._save_plot_image(fig, "sentiment_distribution")
                    register_chart("07_sentiment", img)

            theme_term_frequency = visual_data.get("theme_term_frequency", {}) if isinstance(visual_data, dict) else {}
            if isinstance(theme_term_frequency, dict) and len(theme_term_frequency) > 0:
                labels = []
                values = []
                for k, v in theme_term_frequency.items():
                    try:
                        labels.append(str(k))
                        values.append(float(v))
                    except Exception:
                        continue
                if labels and values:
                    fig, ax = plt.subplots(figsize=(8.2, 3.4))
                    ax.barh(labels, values)
                    ax.set_title("Top Strategic Theme Terms")
                    ax.set_xlabel("Topic Presence Count")
                    ax.invert_yaxis()
                    ax.grid(axis="x", alpha=0.25)
                    img = self._save_plot_image(fig, "theme_term_frequency")
                    register_chart("08_theme_terms", img)

            theme_evolution = visual_data.get("theme_evolution", []) if isinstance(visual_data, dict) else []
            if isinstance(theme_evolution, list) and theme_evolution:
                tmp = []
                for row in theme_evolution:
                    try:
                        tmp.append({
                            "year": int(row.get("year")),
                            "theme": str(row.get("theme")),
                            "weight": float(row.get("weight", 0) or 0),
                        })
                    except Exception:
                        continue
                if tmp:
                    evol_df = pd.DataFrame(tmp)
                    piv = evol_df.pivot_table(index="year", columns="theme", values="weight", aggfunc="mean", fill_value=0.0).sort_index()
                    if len(piv) > 0:
                        fig, ax = plt.subplots(figsize=(8.2, 3.2))
                        for col in piv.columns[:5]:
                            ax.plot(piv.index.astype(int), piv[col].values, marker="o", label=str(col))
                        ax.set_title("Theme Evolution by Year")
                        ax.set_xlabel("Year")
                        ax.set_ylabel("Theme Weight")
                        ax.grid(alpha=0.25)
                        ax.legend(loc="best", fontsize=8)
                        img = self._save_plot_image(fig, "theme_evolution")
                        register_chart("09_theme_evolution", img)

            pulse_daily_counts = visual_data.get("pulse_daily_counts", []) if isinstance(visual_data, dict) else []
            if isinstance(pulse_daily_counts, list) and pulse_daily_counts:
                x = []
                y = []
                for row in pulse_daily_counts:
                    try:
                        x.append(str(row.get("date")))
                        y.append(float(row.get("count", 0) or 0))
                    except Exception:
                        continue
                if x and y:
                    fig, ax = plt.subplots(figsize=(8.2, 3.0))
                    ax.plot(x, y, marker="o")
                    ax.fill_between(x, y, alpha=0.2)
                    ax.set_title("Live Geopolitical Pulse: 7-Day Activity Trend")
                    ax.set_ylabel("Event Count")
                    ax.tick_params(axis="x", rotation=30)
                    ax.grid(axis="y", alpha=0.25)
                    img = self._save_plot_image(fig, "pulse_daily_trend")
                    register_chart("10_pulse_trend", img)

            pulse_recency_split = visual_data.get("pulse_recency_split", {}) if isinstance(visual_data, dict) else {}
            if isinstance(pulse_recency_split, dict) and pulse_recency_split:
                labels = ["Last 24h", "Prior 6d"]
                vals = [
                    float(pulse_recency_split.get("last_24h", 0) or 0),
                    float(pulse_recency_split.get("prior_6d", 0) or 0),
                ]
                if sum(vals) > 0:
                    fig, ax = plt.subplots(figsize=(8.2, 3.0))
                    ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=90, wedgeprops={"width": 0.45})
                    ax.set_title("Live Geopolitical Pulse: Recency Mix")
                    img = self._save_plot_image(fig, "pulse_recency_mix")
                    register_chart("11_pulse_mix", img)

            ordered_keys = [
                "01_strategic_trend",
                "02_confidence",
                "03_quality",
                "04_issue_tags",
                "05_issue_trends",
                "06_tone",
                "07_sentiment",
                "08_theme_terms",
                "09_theme_evolution",
                "10_pulse_trend",
                "11_pulse_mix",
            ]
            images = [chart_by_key[k] for k in ordered_keys if k in chart_by_key]

        except Exception as exc:
            logger.warning(f"Failed to build PDF chart images: {exc}")

        return images
    
    def generate_report(self, 
                       analysis_data: Dict,
                       output_file: str = "diplomatic_barometer_report.pdf",
                       include_visualizations: bool = True) -> bool:
        """
        Generate comprehensive PDF report
        
        Args:
            analysis_data: Dictionary containing analysis results
            output_file: Output PDF file path
            include_visualizations: Whether to include chart images
        
        Returns:
            True if successful, False otherwise
        """
        
        if not REPORTLAB_AVAILABLE:
            logger.error("reportlab is not installed. Cannot generate PDF.")
            return False
        
        chart_images: List[str] = []
        try:
            logger.info(f"Generating PDF report: {output_file}")
            analysis_data = self._normalize_analysis_payload(analysis_data)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_file,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Build document content
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2E86AB'),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#A23B72'),
                spaceAfter=24,
                alignment=TA_CENTER,
                fontName='Helvetica'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2E86AB'),
                spaceAfter=12,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            )
            
            # Add title and subtitle
            story.append(Paragraph(self.title, title_style))
            story.append(Paragraph(self.subtitle, subtitle_style))
            
            # Add metadata
            metadata_text = f"""
            <b>Report Generated:</b> {self.generation_date.strftime('%B %d, %Y at %H:%M:%S')}<br/>
            <b>Author:</b> {self.author}<br/>
            <b>Classification:</b> Research Document
            """
            story.append(Paragraph(metadata_text, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            
            if 'strategic_shift' in analysis_data and analysis_data['strategic_shift']:
                shift_data = analysis_data['strategic_shift']
                
                story.append(Paragraph("<b>Key Findings:</b>", styles['Normal']))
                findings = [
                    f"Total Documents Analyzed: {shift_data.get('total_documents', 'N/A')}",
                    f"Date Range: {shift_data.get('date_range', ['N/A', 'N/A'])[0]} to {shift_data.get('date_range', ['N/A', 'N/A'])[1]}",
                    f"Overall Economic Focus Score: {shift_data.get('overall_economic_avg', 0):.4f}",
                    f"Overall Security Focus Score: {shift_data.get('overall_security_avg', 0):.4f}",
                    f"Dominant Trend: <b>{shift_data.get('trend', 'Unknown')}</b>",
                    f"Strategic Crossover Year: {shift_data.get('crossover_year', 'No crossover detected')}",
                ]
                bullet_style = ParagraphStyle('BulletItem', parent=styles['Normal'], leftIndent=20, bulletIndent=10, bulletFontName='Helvetica', bulletFontSize=10)
                for item in findings:
                    story.append(Paragraph(item, bullet_style, bulletText=chr(8226)))

            # Evidence confidence summary
            if 'confidence_summary' in analysis_data and analysis_data['confidence_summary']:
                conf = analysis_data['confidence_summary']
                story.append(Spacer(1, 0.12*inch))
                story.append(Paragraph("<b>Evidence Confidence:</b>", styles['Normal']))
                conf_items = [
                    f"Confidence Label: {conf.get('label', 'N/A')}",
                    f"Confidence Score: {conf.get('score', 0):.2f}/100",
                ]
                comp = conf.get('components', {}) if isinstance(conf, dict) else {}
                if comp:
                    conf_items.append(
                        "Component Breakdown: "
                        f"sample={comp.get('sample_size', 0):.1f}, "
                        f"source_diversity={comp.get('source_diversity', 0):.1f}, "
                        f"source_credibility={comp.get('source_credibility', 0):.1f}, "
                        f"temporal={comp.get('temporal_coverage', 0):.1f}, "
                        f"stats={comp.get('statistical_strength', 0):.1f}, "
                        f"external={comp.get('external_coverage', 0):.1f}"
                    )
                conf_bullet_style = ParagraphStyle('ConfBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for item in conf_items:
                    story.append(Paragraph(item, conf_bullet_style, bulletText=chr(8226)))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Statistical Significance
            if 'strategic_shift' in analysis_data and 'statistical_significance' in analysis_data['strategic_shift']:
                story.append(Paragraph("Statistical Validation", heading_style))
                
                sig_data = analysis_data['strategic_shift']['statistical_significance']
                
                story.append(Paragraph("<b>Hypothesis Test Results:</b>", styles['Normal']))
                sig_significant = 'Yes - Significant' if sig_data.get('significant') else 'No - Not Significant'
                preferred_p = sig_data.get('preferred_p_value', sig_data.get('p_value', 1))
                preferred_test = sig_data.get('preferred_test', 'paired_t')
                sig_items = [
                    f"Test Used: {preferred_test}",
                    f"T-Statistic: {sig_data.get('t_statistic', 0):.4f}",
                    f"P-Value: {preferred_p:.6f}",
                    f"Statistical Significance (a=0.05): {sig_significant}",
                    f"Effect Size: {sig_data.get('effect_size', 0):.4f}",
                ]
                if 'mean_difference_security_minus_economic' in sig_data:
                    sig_items.append(
                        f"Mean Difference (Security - Economic): {sig_data.get('mean_difference_security_minus_economic', 0):.4f}"
                    )
                if 'ci95_low' in sig_data and 'ci95_high' in sig_data:
                    sig_items.append(
                        f"95% CI: [{sig_data.get('ci95_low', 0):.4f}, {sig_data.get('ci95_high', 0):.4f}]"
                    )
                sig_bullet_style = ParagraphStyle('SigBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for item in sig_items:
                    story.append(Paragraph(item, sig_bullet_style, bulletText=chr(8226)))
            
            story.append(Spacer(1, 0.2*inch))

            if include_visualizations:
                chart_images = self._build_chart_images(analysis_data)
                if chart_images:
                    story.append(Paragraph("Key Visualizations", heading_style))
                    for img_path in chart_images[:11]:
                        try:
                            story.append(Image(img_path, width=7.0 * inch, height=2.9 * inch))
                            story.append(Spacer(1, 0.12 * inch))
                        except Exception:
                            continue
                else:
                    if not MATPLOTLIB_AVAILABLE:
                        story.append(Paragraph("Visualization note: matplotlib is not available in the current environment, so charts are omitted.", styles['Italic']))
                    else:
                        story.append(Paragraph("Visualization note: chart-ready series were not present in the analysis payload, so charts are omitted.", styles['Italic']))
            
            # Trend Analysis
            if 'strategic_shift' in analysis_data and 'trend_analysis' in analysis_data['strategic_shift']:
                story.append(Paragraph("Trend Analysis", heading_style))
                
                trend_data = analysis_data['strategic_shift']['trend_analysis']
                
                story.append(Paragraph("<b>Linear Regression Analysis (per year):</b>", styles['Normal']))
                trend_items = [
                    f"Economic Focus Trend: {trend_data.get('economic_slope', 0):.6f} ({trend_data.get('econ_direction', 'N/A')})",
                    f"Security Focus Trend: {trend_data.get('security_slope', 0):.6f} ({trend_data.get('sec_direction', 'N/A')})",
                ]
                trend_bullet_style = ParagraphStyle('TrendBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for item in trend_items:
                    story.append(Paragraph(item, trend_bullet_style, bulletText=chr(8226)))
            
            story.append(PageBreak())
            
            # Tone Analysis Results
            if 'tone_analysis' in analysis_data:
                story.append(Paragraph("Tone & Sentiment Analysis", heading_style))
                
                tone_data = analysis_data['tone_analysis']
                story.append(Paragraph("<b>Tone Distribution:</b>", styles['Normal']))
                tone_bullet_style = ParagraphStyle('ToneBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                story.append(Paragraph(f"Total Documents Analyzed: {tone_data.get('total_documents', 'N/A')}", tone_bullet_style, bulletText=chr(8226)))
                
                if 'tone_distribution' in tone_data:
                    for tone, count in tone_data['tone_distribution'].items():
                        story.append(Paragraph(f"{tone}: {count}", tone_bullet_style, bulletText=chr(8226)))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Thematic Analysis Results
            if 'thematic_analysis' in analysis_data:
                story.append(Paragraph("Thematic Analysis", heading_style))
                
                thematic_data = analysis_data['thematic_analysis']
                story.append(Paragraph("<b>Topic Modeling Results:</b>", styles['Normal']))
                thematic_bullet_style = ParagraphStyle('ThematicBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                story.append(Paragraph(f"Number of Topics Discovered: {thematic_data.get('num_topics', 'N/A')}", thematic_bullet_style, bulletText=chr(8226)))
                story.append(Paragraph(f"Themes: {', '.join(map(str, thematic_data.get('themes', [])))}", thematic_bullet_style, bulletText=chr(8226)))

            visual_data = analysis_data.get('visual_data', {}) if isinstance(analysis_data, dict) else {}
            issue_rows = visual_data.get('issue_tag_counts', []) if isinstance(visual_data, dict) else []
            if isinstance(issue_rows, list) and issue_rows:
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph("Issue Tag Landscape", heading_style))
                issue_bullet_style = ParagraphStyle('IssueBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for row in issue_rows[:8]:
                    issue = str(row.get('issue', ''))
                    docs = row.get('documents', 0)
                    story.append(Paragraph(f"{issue}: {docs} documents", issue_bullet_style, bulletText=chr(8226)))

            issue_trends = visual_data.get('issue_tag_trends', []) if isinstance(visual_data, dict) else []
            if isinstance(issue_trends, list) and issue_trends:
                story.append(Spacer(1, 0.12*inch))
                story.append(Paragraph("Issue Tag Trends", heading_style))
                trend_bullet_style = ParagraphStyle('IssueTrendBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                try:
                    tdf = pd.DataFrame(issue_trends)
                    if {'issue', 'documents'}.issubset(set(tdf.columns)):
                        top = tdf.groupby('issue', as_index=False)['documents'].sum().sort_values('documents', ascending=False).head(5)
                        for _, row in top.iterrows():
                            story.append(Paragraph(f"{row['issue']}: {row['documents']:.0f} total mentions across years", trend_bullet_style, bulletText=chr(8226)))
                except Exception:
                    pass
            
            story.append(Spacer(1, 0.3*inch))
            
            # Methodology section
            story.append(Paragraph("Methodology", heading_style))
            
            methodology = """
            <b>Data Sources:</b> Ministry of External Affairs (MEA) India and Ministry of Foreign Affairs (MOFA) Japan<br/>
            <b>Analysis Period:</b> 2000-2025<br/>
            <b>NLP Techniques:</b> Lemmatization, Named Entity Recognition, Lexicon-based categorization, Topic Modeling (LDA)<br/>
            <b>Statistical Methods:</b> Independent t-tests, Linear regression, Effect size calculation<br/>
            <b>Key Lexicons:</b> Expanded economic and security terminology sets<br/>
            """
            
            story.append(Paragraph(methodology, styles['Normal']))

            # Action-ready policy sections
            if 'policy_triggers' in analysis_data and analysis_data['policy_triggers']:
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph("Policy Trigger Alerts", heading_style))
                trigger_bullet_style = ParagraphStyle('TriggerBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for row in analysis_data['policy_triggers'][:12]:
                    item = (
                        f"[{row.get('severity', 'Info')}] {row.get('trigger', '')}: "
                        f"{row.get('condition', '')} | Action: {row.get('policy_action', '')}"
                    )
                    story.append(Paragraph(item, trigger_bullet_style, bulletText=chr(8226)))

            if 'decision_options' in analysis_data and analysis_data['decision_options']:
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph("Decision Option Cards", heading_style))
                option_bullet_style = ParagraphStyle('OptionBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for opt in analysis_data['decision_options'][:10]:
                    item = (
                        f"{opt.get('option', '')} ({opt.get('risk_level', 'Info')}): "
                        f"When: {opt.get('when_to_use', '')} | "
                        f"Action: {opt.get('recommended_action', '')} | "
                        f"Evidence: {opt.get('evidence_anchor', '')}"
                    )
                    story.append(Paragraph(item, option_bullet_style, bulletText=chr(8226)))

            if 'quality_audit' in analysis_data and analysis_data['quality_audit']:
                qa = analysis_data['quality_audit']
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph("Quarterly Quality Audit", heading_style))
                qa_items = [
                    f"Audit Score: {qa.get('audit_score', 0):.1f}/100",
                    f"Status: {qa.get('status', 'Unknown')}",
                    f"Quarters Analyzed: {qa.get('quarters_analyzed', 0)}",
                    f"Issues Count: {qa.get('issues_count', 0)}",
                ]
                qa_bullet_style = ParagraphStyle('QABullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for item in qa_items:
                    story.append(Paragraph(item, qa_bullet_style, bulletText=chr(8226)))

            if 'live_pulse' in analysis_data and analysis_data['live_pulse']:
                lp = analysis_data['live_pulse']
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph("Live Geopolitical Pulse", heading_style))
                lp_items = [
                    f"Events (24h): {lp.get('count_24h', 0)}",
                    f"Events (7d): {lp.get('count_7d', 0)}",
                    f"Providers (7d): {lp.get('providers', 0)}",
                    f"Pulse Confidence: {lp.get('confidence_label', 'Low')} ({lp.get('confidence_score', 0):.1f}/100)",
                ]
                lp_bullet_style = ParagraphStyle('PulseBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for item in lp_items:
                    story.append(Paragraph(item, lp_bullet_style, bulletText=chr(8226)))

                top_events = lp.get('top_events', []) if isinstance(lp, dict) else []
                if isinstance(top_events, list) and len(top_events) > 0:
                    story.append(Paragraph("Top events (7d):", styles['Normal']))
                    evt_style = ParagraphStyle('PulseEvent', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                    for row in top_events[:5]:
                        evt = (
                            f"[{row.get('provider', '')}] {row.get('title', '')} "
                            f"({row.get('published_at', '')})"
                        )
                        story.append(Paragraph(evt, evt_style, bulletText=chr(8226)))
            
            # Footer
            story.append(Spacer(1, 0.5*inch))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
            story.append(Paragraph(
                "This is an automated research report generated by the Cross-Lingual NLP Framework.",
                footer_style
            ))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated successfully: {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            return False
        finally:
            for p in chart_images:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
    
    def generate_summary_pdf(self,
                            summary_text: str,
                            output_file: str = "diplomatic_barometer_summary.pdf") -> bool:
        """Generate a simple summary PDF"""
        
        if not REPORTLAB_AVAILABLE:
            logger.error("reportlab is not installed")
            return False
        
        try:
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            story.append(Paragraph(self.title, styles['Title']))
            story.append(Paragraph(self.subtitle, styles['Heading2']))
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph(summary_text, styles['BodyText']))
            
            doc.build(story)
            logger.info(f"Summary PDF generated: {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error generating summary PDF: {str(e)}")
            return False
    
    def export_to_pdf(self, analysis_dict: Dict, filename: str = None) -> str:
        """Export analysis dictionary to PDF report"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"diplomatic_barometer_report_{timestamp}.pdf"
        
        success = self.generate_report(analysis_dict, filename)
        
        if success:
            logger.info(f"Report exported to: {filename}")
            return filename
        else:
            logger.error(f"Failed to export PDF: {filename}")
            return None


def demo_pdf_generation():
    """Demo PDF generation"""
    
    sample_analysis = {
        'strategic_shift': {
            'total_documents': 10,
            'date_range': ['2005-01-01', '2024-12-01'],
            'overall_economic_avg': 0.2871,
            'overall_security_avg': 0.2011,
            'crossover_year': 2022,
            'trend': 'ECONOMIC',
            'statistical_significance': {
                't_statistic': -1.234,
                'p_value': 0.2345,
                'significant': False,
                'effect_size': -0.456
            },
            'trend_analysis': {
                'economic_slope': -0.001234,
                'econ_direction': 'Decreasing',
                'security_slope': 0.002345,
                'sec_direction': 'Increasing'
            }
        },
        'tone_analysis': {
            'total_documents': 10,
            'tone_distribution': {
                'Formal': 6,
                'Assertive': 3,
                'Emphatic': 1
            }
        },
        'thematic_analysis': {
            'num_topics': 5,
            'themes': [0, 1, 2, 3, 4]
        }
    }
    
    generator = PDFReportGenerator()
    output_file = generator.export_to_pdf(sample_analysis)
    
    if output_file:
        print(f"PDF report generated: {output_file}")
    else:
        print("PDF generation requires reportlab: pip install reportlab")


if __name__ == "__main__":
    demo_pdf_generation()
