"""
PDF report generation using ReportLab
"""

import json
from datetime import datetime
from typing import Dict, Optional, List
import logging

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


class PDFReportGenerator:
    """Generate comprehensive PDF reports from analysis data"""
    
    def __init__(self, title: str = "Cross-Lingual NLP Framework", subtitle: str = "India-Japan Strategic Shift Analysis"):
        self.title = title
        self.subtitle = subtitle
        self.author = "Cross-Lingual NLP Framework Team"
        self.generation_date = datetime.now()
    
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
        
        try:
            logger.info(f"Generating PDF report: {output_file}")
            
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
            
            story.append(Spacer(1, 0.2*inch))
            
            # Statistical Significance
            if 'strategic_shift' in analysis_data and 'statistical_significance' in analysis_data['strategic_shift']:
                story.append(Paragraph("Statistical Validation", heading_style))
                
                sig_data = analysis_data['strategic_shift']['statistical_significance']
                
                story.append(Paragraph("<b>Hypothesis Test Results:</b>", styles['Normal']))
                sig_significant = 'Yes - Significant' if sig_data.get('significant') else 'No - Not Significant'
                sig_items = [
                    f"T-Statistic: {sig_data.get('t_statistic', 0):.4f}",
                    f"P-Value: {sig_data.get('p_value', 1):.6f}",
                    f"Statistical Significance (a=0.05): {sig_significant}",
                    f"Effect Size (Cohen's d): {sig_data.get('effect_size', 0):.4f}",
                ]
                sig_bullet_style = ParagraphStyle('SigBullet', parent=styles['Normal'], leftIndent=20, bulletIndent=10)
                for item in sig_items:
                    story.append(Paragraph(item, sig_bullet_style, bulletText=chr(8226)))
            
            story.append(Spacer(1, 0.2*inch))
            
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
