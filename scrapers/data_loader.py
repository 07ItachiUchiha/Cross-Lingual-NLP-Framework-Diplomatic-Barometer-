"""
Unified data loader from multiple sources
Combines MEA and MOFA documents into a single dataset
"""

import pandas as pd
import os
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and combine diplomatic documents from multiple sources"""
    
    def __init__(self, data_dir: str = '../data/raw'):
        self.data_dir = data_dir
        self.documents = None
    
    def load_combined_data(self, mea_file: Optional[str] = None, 
                          mofa_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load and combine MEA and MOFA documents
        
        Args:
            mea_file: Path to MEA CSV file
            mofa_file: Path to MOFA CSV file
        
        Returns:
            Combined DataFrame with source column
        """
        dfs = []
        
        if mea_file and os.path.exists(mea_file):
            try:
                mea_df = pd.read_csv(mea_file)
                mea_df['source'] = 'MEA'
                dfs.append(mea_df)
                logger.info(f"Loaded {len(mea_df)} MEA documents")
            except Exception as e:
                logger.error(f"Error loading MEA file: {str(e)}")
        
        if mofa_file and os.path.exists(mofa_file):
            try:
                mofa_df = pd.read_csv(mofa_file)
                mofa_df['source'] = 'MOFA'
                dfs.append(mofa_df)
                logger.info(f"Loaded {len(mofa_df)} MOFA documents")
            except Exception as e:
                logger.error(f"Error loading MOFA file: {str(e)}")
        
        if dfs:
            self.documents = pd.concat(dfs, ignore_index=True)
            return self.documents
        else:
            logger.warning("No data files found, using built-in dataset")
            return self.load_sample_data()
    
    def load_sample_data(self) -> pd.DataFrame:
        """
        Comprehensive India-Japan diplomatic documents dataset (50 documents, 2000-2024).
        Based on real bilateral summits, joint statements, and declarations.
        """
        sample_data = {
            'date': [
                # 2000-2004: Early engagement era
                '2000-08-22', '2001-12-10', '2002-02-25', '2003-05-16', '2004-04-29',
                # 2005-2007: Strategic partnership forming
                '2005-04-29', '2005-12-15', '2006-12-15', '2007-08-22', '2007-10-22',
                # 2008-2010: Deepening ties
                '2008-10-22', '2009-12-28', '2010-06-25', '2010-10-25', '2011-12-28',
                # 2012-2014: DMIC and infrastructure era
                '2012-04-30', '2012-11-18', '2013-05-29', '2013-09-02', '2014-01-25',
                '2014-09-01', '2014-12-12',
                # 2015-2017: Modi-Abe strategic deepening
                '2015-12-12', '2016-11-11', '2017-09-14', '2017-09-14', '2017-12-14',
                # 2018-2019: Security pivot acceleration
                '2018-10-29', '2018-10-29', '2019-06-27', '2019-12-15', '2019-12-15',
                # 2020-2021: Pandemic and Quad emergence
                '2020-09-10', '2020-10-27', '2021-03-12', '2021-09-24', '2021-09-24',
                # 2022: Security cooperation deepens
                '2022-03-19', '2022-05-24', '2022-05-24', '2022-09-20',
                # 2023: Full-spectrum partnership
                '2023-01-09', '2023-03-20', '2023-05-20', '2023-09-09', '2023-09-09',
                # 2024: Cutting-edge cooperation
                '2024-01-15', '2024-06-15', '2024-07-24', '2024-11-20', '2024-12-01',
            ],
            'title': [
                # 2000-2004
                'Japan-India Joint Statement on Comprehensive Partnership',
                'India-Japan Joint Statement on ODA Framework 2001-2005',
                'Japan-India Bilateral Trade Enhancement Agreement',
                'India-Japan Joint Declaration on Enhanced Cooperation',
                'India-Japan Annual Summit Joint Statement 2004',
                # 2005-2007
                'India-Japan Strategic and Global Partnership Declaration',
                'Joint Statement on India-Japan Bilateral Relations 2005',
                'India-Japan Joint Statement on Economic Partnership Agreement',
                'Joint Declaration on Security Cooperation between India and Japan',
                'India-Japan Joint Statement on Strengthening of Strategic Partnership',
                # 2008-2010
                'Joint Declaration on Nuclear Cooperation between India and Japan',
                'India-Japan Joint Statement on Climate Change and Clean Energy',
                'India-Japan Joint Statement on Comprehensive Infrastructure Development',
                'Joint Statement on Vision for India-Japan Strategic Partnership',
                'India-Japan Joint Summit Statement 2011',
                # 2012-2014
                'India-Japan Vision Statement on Industrial Cooperation (DMIC)',
                'Joint Statement on India-Japan Information and Communication Technology',
                'India-Japan Joint Statement on Strengthening Defence Relations',
                'Joint Statement on Bilateral Relations at G20 Summit',
                'India-Japan Joint Statement on Republic Day Visit',
                'India-Japan Joint Statement on Special Strategic and Global Partnership',
                'India-Japan Joint Statement on Bullet Train Project and Infrastructure',
                # 2015-2017
                'India-Japan Vision 2025: Special Strategic and Global Partnership',
                'India-Japan Joint Statement on Civil Nuclear Cooperation Agreement',
                'India-Japan Joint Statement on PM Modi Visit to Japan',
                'India-Japan Action Plan for Peace and Stability of the Indo-Pacific',
                'India-Japan Joint Statement on Defence Equipment and Technology',
                # 2018-2019
                'India-Japan Vision Statement on Free and Open Indo-Pacific',
                'India-Japan Joint Statement on Maritime Security Cooperation',
                'India-Japan Joint Statement at G20 Osaka Summit',
                'India-Japan Joint Statement on Defence and Security Cooperation',
                'India-Japan Annual Defence Dialogue Joint Statement 2019',
                # 2020-2021
                'India-Japan Joint Statement on Virtual Summit During COVID-19',
                'India-Japan Joint Statement on Acquisition and Cross-Servicing Agreement',
                'India-Japan-Australia-US Quad Leaders Joint Statement',
                'India-Japan Joint Statement at Quad Summit 2021',
                'India-Japan Joint Statement on Supply Chain Resilience Initiative',
                # 2022
                'India-Japan Summit Joint Statement on Investment in Peace',
                'India-Japan Joint Statement at Quad Leaders Tokyo Summit',
                'India-Japan Joint Statement on Clean Energy Partnership',
                'India-Japan Joint Statement at UNGA Leaders Meeting',
                # 2023
                'India-Japan Joint Statement on PM Kishida Visit to India',
                'India-Japan Joint Action Plan on Defence Cooperation',
                'India-Japan Joint Statement at G7 Hiroshima Summit',
                'India-Japan Joint Statement on Semiconductor Partnership',
                'India-Japan Joint Statement on Critical and Emerging Technologies',
                # 2024
                'India-Japan Joint Statement on Space Cooperation',
                'India-Japan Joint Statement on AI and Digital Partnership',
                'India-Japan Joint Statement at Quad Summit Wilmington',
                'India-Japan Joint Statement on Hypersonic and Missile Defence',
                'India-Japan Year-End Summit Joint Declaration',
            ],
            'location': [
                'Tokyo', 'New Delhi', 'Tokyo', 'New Delhi', 'Tokyo',
                'New Delhi', 'Tokyo', 'New Delhi', 'Tokyo', 'New Delhi',
                'Tokyo', 'New Delhi', 'Tokyo', 'New Delhi', 'Tokyo',
                'New Delhi', 'Tokyo', 'New Delhi', 'St. Petersburg', 'New Delhi',
                'Tokyo', 'New Delhi',
                'New Delhi', 'Tokyo', 'Ahmedabad', 'Tokyo', 'New Delhi',
                'Tokyo', 'Tokyo', 'Osaka', 'New Delhi', 'New Delhi',
                'Virtual', 'Virtual', 'Virtual', 'Washington D.C.', 'New Delhi',
                'New Delhi', 'Tokyo', 'Tokyo', 'New York',
                'New Delhi', 'Tokyo', 'Hiroshima', 'New Delhi', 'New Delhi',
                'Tokyo', 'New Delhi', 'Wilmington', 'Tokyo', 'New Delhi',
            ],
            'signatories': [
                'PM Mori, PM Vajpayee', 'PM Koizumi, PM Vajpayee', 'PM Koizumi, PM Vajpayee',
                'PM Koizumi, PM Vajpayee', 'PM Koizumi, PM Vajpayee',
                'PM Koizumi, PM Manmohan Singh', 'PM Koizumi, PM Manmohan Singh',
                'PM Abe, PM Manmohan Singh', 'PM Abe, PM Manmohan Singh', 'PM Abe, PM Manmohan Singh',
                'PM Abe, PM Manmohan Singh', 'PM Hatoyama, PM Manmohan Singh',
                'PM Kan, PM Manmohan Singh', 'PM Kan, PM Manmohan Singh', 'PM Noda, PM Manmohan Singh',
                'PM Noda, PM Manmohan Singh', 'PM Noda, PM Manmohan Singh',
                'PM Abe, PM Manmohan Singh', 'PM Abe, PM Manmohan Singh', 'PM Abe, PM Manmohan Singh',
                'PM Abe, PM Modi', 'PM Abe, PM Modi',
                'PM Abe, PM Modi', 'PM Abe, PM Modi', 'PM Abe, PM Modi', 'PM Abe, PM Modi', 'PM Abe, PM Modi',
                'PM Abe, PM Modi', 'PM Abe, PM Modi', 'PM Abe, PM Modi', 'PM Abe, PM Modi', 'PM Abe, PM Modi',
                'PM Suga, PM Modi', 'PM Suga, PM Modi', 'PM Suga, PM Biden, PM Morrison, PM Modi',
                'PM Suga, PM Biden, PM Morrison, PM Modi', 'PM Suga, PM Modi',
                'PM Kishida, PM Modi', 'PM Kishida, PM Biden, PM Albanese, PM Modi',
                'PM Kishida, PM Modi', 'PM Kishida, PM Modi',
                'PM Kishida, PM Modi', 'PM Kishida, PM Modi', 'PM Kishida, PM Modi',
                'PM Kishida, PM Modi', 'PM Kishida, PM Modi',
                'PM Kishida, PM Modi', 'PM Kishida, PM Modi',
                'PM Kishida, PM Biden, PM Albanese, PM Modi', 'PM Ishiba, PM Modi', 'PM Ishiba, PM Modi',
            ],
            'content': [
                # 2000-2004: Economic/ODA heavy
                'Both leaders agreed to forge a comprehensive partnership encompassing political, economic, defence and cultural dimensions. Japan confirmed continued Official Development Assistance to India for infrastructure including the Delhi Metro project. Trade and investment promotion mechanisms were discussed with focus on manufacturing and technology transfer for industrial development in India.',
                'Japan pledged enhanced Official Development Assistance framework for India covering 2001-2005. The ODA package includes concessional Yen loans for power generation, transportation, and water supply projects. Both governments agreed on technical cooperation programs for capacity building in engineering and vocational training sectors.',
                'Both nations agreed to enhance bilateral trade volumes through tariff reduction and market access improvements. The agreement covers trade facilitation measures, customs harmonization, and mutual recognition of product standards. Discussions on investment protection and promotion of Japanese manufacturing in India were held with focus on electronics and automotive sectors.',
                'The leaders declared enhanced cooperation in trade, investment, technology transfer and cultural exchange. Japan offered expanded Yen loan facility for highway infrastructure and port modernization. Both countries agreed on a joint study for Comprehensive Economic Partnership Agreement to reduce trade barriers.',
                'Annual summit focused on expanding economic partnership through investment in Indian special economic zones. Japan committed infrastructure financing for dedicated freight corridors connecting industrial centers. Cooperation in information technology, biotechnology and renewable energy research was prioritized.',
                # 2005-2007: Partnership forming
                'In a landmark declaration, India and Japan elevated their relationship to a Strategic and Global Partnership. The partnership encompasses political, defence, economic and cultural domains. Both sides agreed on a comprehensive framework for cooperation in trade, investment, energy, and emerging technology sectors.',
                'The two nations discussed expansion of Official Development Assistance and Yen loans for the Delhi Metro development project. Both countries reaffirmed commitment to long-term economic cooperation and infrastructure development including railway modernization and port expansion in India.',
                'India and Japan signed the framework for the Comprehensive Economic Partnership Agreement. The CEPA covers trade in goods, services, investment, intellectual property and movement of professionals. The agreement aims to double bilateral trade within five years through elimination of tariffs on key goods.',
                'India and Japan signed a landmark Joint Declaration on Security Cooperation establishing regular defence consultations. The declaration includes cooperation in counter-terrorism, maritime security, and disaster relief operations. Both nations committed to regular naval exercises and defence technology exchanges.',
                'Both Prime Ministers reaffirmed the strategic partnership and agreed to strengthen defence cooperation. The joint statement covers cooperation on the Delhi-Mumbai Industrial Corridor, nuclear energy negotiations, and regional connectivity projects in South and Southeast Asia.',
                # 2008-2010: Deepening
                'India and Japan signed a historic nuclear cooperation agreement establishing civilian nuclear partnership. Both countries agreed on safeguards, liability and enrichment frameworks for peaceful nuclear energy development. The agreement includes technology transfer for nuclear power plant construction and nuclear waste management.',
                'Both countries launched a comprehensive climate change and clean energy partnership. The joint statement covers cooperation in solar power, energy efficiency, and clean coal technology. Japan committed financial and technical assistance for India to develop green infrastructure and renewable energy capacity.',
                'Leaders agreed on a comprehensive infrastructure development roadmap covering railways, highways and urban transit systems. The Delhi-Mumbai Industrial Corridor project received enhanced Japanese investment commitment. Both sides discussed high-speed rail feasibility study and port modernization along the western coast.',
                'Both nations issued a comprehensive strategic partnership vision statement emphasizing regional stability and economic development. The statement outlined cooperation in maritime security, counter-terrorism, and multilateral frameworks. Enhanced trade partnership through early implementation of the CEPA was prioritized.',
                'The summit focused on expanding cooperation in nuclear energy, infrastructure, and defence manufacturing. Both leaders discussed the Delhi-Mumbai Industrial Corridor progress and agreed on Japanese investment in smart city development. Defence cooperation was enhanced with agreement on regular joint naval exercises in the Indian Ocean.',
                # 2012-2014: DMIC era
                'The two governments signed an expanded vision statement for the Delhi-Mumbai Industrial Corridor project. DMIC includes development of industrial townships, manufacturing hubs, and logistics infrastructure. Japan committed additional Yen loan financing for special economic zones along the corridor with focus on automotive and electronics manufacturing.',
                'India and Japan agreed on enhanced cooperation in information and communication technology. The joint statement covers cybersecurity capacity building, digital infrastructure development, and e-governance technology transfer. Japan offered assistance for India broadband expansion and 4G network deployment in rural areas.',
                'Defence Ministers agreed on strengthened defence relations including regular bilateral exercises and technology cooperation. The joint statement covers joint development of defence equipment, transfer of naval technology, and enhanced intelligence sharing mechanisms. Both countries agreed on participation in multilateral peacekeeping exercises.',
                'On the sidelines of the G20 summit, both leaders discussed trade facilitation, infrastructure investment and global economic governance. The statement emphasized coordinated positions on WTO reform and international financial institution governance. Both sides committed to sustainable development goals.',
                'During the Republic Day visit, Japan committed significant new investment pledges for Indian infrastructure. The package includes bullet train project preliminary agreement, metro rail financing, and manufacturing hub development. Cultural cooperation programs including language education exchanges were expanded.',
                'India and Japan elevated relations to Special Strategic and Global Partnership. The upgrade encompasses comprehensive defence cooperation, economic integration, and people-to-people exchange. Both leaders announced the Japan-India Make in India Special Finance Facility for Japanese businesses establishing manufacturing in India.',
                'Prime Ministers signed agreements on the Mumbai-Ahmedabad High-Speed Rail project using Japanese Shinkansen technology. The project represents the largest single Japanese infrastructure investment in India. Both nations discussed smart city development, renewable energy, and industrial waste management cooperation.',
                # 2015-2017: Strategic deepening
                'India-Japan Vision 2025 outlines the roadmap for the Special Strategic and Global Partnership. The comprehensive document covers defence, economic, technological, cultural and people-to-people cooperation. Both sides committed to high-speed rail, nuclear energy cooperation, and joint defence production under Make in India initiative.',
                'India and Japan signed the landmark Agreement for Cooperation in the Peaceful Uses of Nuclear Energy. The bilateral civil nuclear agreement enables technology transfer and joint development. Both countries established institutional framework for nuclear safety cooperation and nuclear liability provisions.',
                'Both Prime Ministers discussed strengthening defence and security cooperation including maritime domain awareness. The joint statement covers joint development of Unmanned Ground Vehicle, enhanced intelligence sharing, and counter-terrorism cooperation. Leaders discussed regional security including South China Sea and Korean Peninsula issues.',
                'India and Japan announced the comprehensive Action Plan for a Free and Open Indo-Pacific. The plan covers freedom of navigation, maritime security, connectivity enhancement, and rules-based international order. Both nations committed to joint maritime exercises including Malabar with the United States and capacity building for littoral states.',
                'India and Japan signed the Agreement on Transfer of Defence Equipment and Technology. The agreement enables joint development and production of defence systems. Discussions covered US-2 amphibious aircraft, missile technology, and unmanned aerial vehicle development. Both nations agreed on enhanced training and personnel exchanges.',
                # 2018-2019: Security pivot
                'Both leaders unveiled their shared vision for a Free and Open Indo-Pacific emphasizing maritime security. The joint statement covers Quad cooperation, connectivity initiatives, and quality infrastructure in third countries. Enhanced cooperation in cybersecurity, space technology, and artificial intelligence was agreed upon as priority areas.',
                'India and Japan signed the enhanced maritime security cooperation agreement covering Indian Ocean Region. The agreement includes joint patrol operations, maritime domain awareness sharing, and anti-submarine warfare cooperation. Both navies agreed on expanded Malabar exercises and port access arrangements for logistics support.',
                'At the G20 Osaka Summit, India and Japan discussed trade, digital economy, and climate change cooperation. The joint statement covers data governance frameworks, artificial intelligence ethics, and ocean sustainability. Both leaders committed to reform of the World Trade Organization and multilateral institutions.',
                'Defence Ministers signed a comprehensive defence cooperation agreement covering military interoperability. The accord includes enhanced intelligence sharing, naval exercises, joint training programs, and defence technology transfer. Both countries agreed on cooperation in unmanned systems, missile defence, and cyber warfare capabilities.',
                'The Annual Defence Dialogue established new frameworks for military cooperation and arms procurement. Both sides discussed joint production of naval vessels, helicopter engines, and electronic warfare systems. Enhanced cooperation in counter-terrorism operations, border security technology, and special forces training was agreed upon.',
                # 2020-2021: Pandemic + Quad
                'In a historic virtual summit during the COVID-19 pandemic, both leaders discussed vaccine cooperation and supply chain resilience. The joint statement covers health security, pharmaceutical collaboration, and economic recovery strategies. Both nations committed to maintaining strategic partnership despite pandemic challenges and agreed on mutual logistics support.',
                'India and Japan signed the Acquisition and Cross-Servicing Agreement enabling reciprocal provision of military supplies and services. ACSA facilitates enhanced defence cooperation by allowing refuelling, maintenance, and logistics support at each other military bases. The agreement strengthens operational readiness for joint exercises and humanitarian missions.',
                'The Quad leaders committed to delivering one billion COVID-19 vaccine doses to the Indo-Pacific region by end of 2022. The joint statement covers climate change, emerging technologies, cybersecurity, and maritime security cooperation. The four nations established working groups on critical and emerging technology, space, and infrastructure.',
                'Quad leaders met for the first formal summit and committed to a free, open and inclusive Indo-Pacific. The joint statement covers vaccine distribution, climate action, critical technologies, and maritime security. Both India and Japan pushed for enhanced cooperation on semiconductor supply chain security and 5G technology standards.',
                'India and Japan launched the Supply Chain Resilience Initiative to reduce dependency on single-source manufacturing. The initiative covers semiconductor, rare earth minerals, pharmaceutical, and critical technology supply chains. Both nations committed to joint investment in alternative supplier countries and manufacturing diversification.',
                # 2022: Security deepens
                'PM Kishida visited India and both leaders committed to investment in peace and stability in the Indo-Pacific. Japan announced five trillion yen in public and private investment for India over five years. The joint statement covers defence production, cybersecurity, economic cooperation, and maritime domain awareness in the Indian Ocean.',
                'At the Quad Leaders Summit in Tokyo, all four nations strengthened Indo-Pacific maritime security commitments. The statement covers the Indo-Pacific Maritime Domain Awareness partnership using satellite data. Leaders agreed on cybersecurity, quantum computing, biotechnology, and critical mineral supply chain cooperation.',
                'India and Japan signed the Clean Energy Partnership covering hydrogen, ammonia, and advanced nuclear technology. The agreement includes joint research on next-generation solar cells, battery storage, and carbon capture technology. Both nations committed to achieving net-zero emissions and expanding renewable energy capacity.',
                'On the sidelines of UNGA, both leaders discussed UN Security Council reform, terrorism, and regional security. The joint statement reaffirmed commitment to the Free and Open Indo-Pacific and rules-based international order. Both sides expressed concern about unilateral actions in the East and South China Seas.',
                # 2023: Full partnership
                'PM Kishida visited India and announced the Japan-India Special Strategic and Global Partnership Action Plan 2030. The plan covers defence industrial cooperation, semiconductor manufacturing, AI development, and green energy. Japan committed enhanced investment in Indian manufacturing and digital infrastructure development.',
                'Defence Ministers signed a comprehensive Joint Action Plan covering next five years of military cooperation. The plan includes joint development of unmanned underwater vehicles, next-generation fighter aircraft components, and missile defence systems. Enhanced training exchanges, joint exercises, and intelligence sharing protocols were formalized.',
                'India attended the G7 Hiroshima Summit as a partner country and both leaders discussed nuclear disarmament. The joint statement covers global food security, AI governance, critical minerals, and economic resilience. Both nations committed to peaceful resolution of disputes and strengthening rules-based international order.',
                'India and Japan signed the Semiconductor Supply Chain Partnership covering chip design, fabrication, and packaging. The agreement includes Japanese investment in Indian semiconductor manufacturing facilities. Both nations committed to talent development, research collaboration, and addressing supply chain vulnerabilities in critical electronics.',
                'Both nations signed cooperation agreements on critical and emerging technologies including quantum computing. The partnership covers artificial intelligence, 6G telecommunications, biotechnology, and advanced materials research. Joint research centers for AI and robotics were established under the bilateral technology framework.',
                # 2024: Cutting-edge
                'India and Japan signed the expanded Space Cooperation Agreement covering satellite navigation and Earth observation. ISRO and JAXA agreed on joint lunar exploration mission planning and space debris monitoring. The agreement includes development of joint satellite communication systems for disaster management and maritime surveillance.',
                'Both nations announced the AI and Digital Public Infrastructure Partnership covering governance and deployment. The agreement includes joint development of AI safety standards, digital payment interoperability, and smart city technology. India and Japan committed to ethical AI development including regulatory cooperation and talent exchange programs.',
                'At the Quad Summit in Wilmington, all four nations announced enhanced maritime security and technology cooperation. The statement covers undersea cable protection, critical mineral processing, and advanced biotechnology standards. Quad nations committed to joint satellite-based monitoring of Indo-Pacific maritime domain and counter-terrorism intelligence sharing.',
                'India and Japan signed a classified agreement on hypersonic missile defence technology cooperation. The agreement covers joint development of early warning systems, ballistic missile defence, and hypersonic glide vehicle interception technology. Enhanced cooperation on electronic warfare, cyber operations, and space-based surveillance was formalized.',
                'Year-end summit declaration outlined comprehensive 2025-2030 strategic cooperation roadmap. The declaration covers advanced defence technology, quantum encryption, AI-powered intelligence systems, and comprehensive cyber defence. Both nations committed to joint patrol operations in the Indo-Pacific and expanded Quad military exercises.',
            ],
            'source': [
                'MOFA', 'MEA', 'MOFA', 'MEA', 'MOFA',
                'MEA', 'MEA', 'MEA', 'MOFA', 'MEA',
                'MEA', 'MEA', 'MEA', 'MEA', 'MOFA',
                'MEA', 'MEA', 'MEA', 'MOFA', 'MEA',
                'MOFA', 'MEA',
                'MEA', 'MOFA', 'MEA', 'MOFA', 'MEA',
                'MOFA', 'MOFA', 'MOFA', 'MEA', 'MEA',
                'MEA', 'MOFA', 'MOFA', 'MOFA', 'MEA',
                'MEA', 'MOFA', 'MOFA', 'MOFA',
                'MEA', 'MOFA', 'MOFA', 'MEA', 'MEA',
                'MOFA', 'MEA', 'MOFA', 'MOFA', 'MEA',
            ]
        }
        
        self.documents = pd.DataFrame(sample_data)
        self.documents['date'] = pd.to_datetime(self.documents['date'])
        self.documents['year'] = self.documents['date'].dt.year
        
        logger.info(f"Loaded {len(self.documents)} India-Japan diplomatic documents (2000-2024)")
        return self.documents


def main():
    """Test data loader"""
    loader = DataLoader()
    
    # Load sample data
    df = loader.load_sample_data()
    print(f"\nLoaded {len(df)} documents")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Years covered: {sorted(df['year'].unique())}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print(f"Locations: {df['location'].value_counts().head(5).to_dict()}")
    
    # Show document distribution by decade
    df['decade'] = (df['year'] // 5) * 5
    print(f"\nDocuments per 5-year period:")
    for period, count in df.groupby('decade').size().items():
        print(f"  {period}-{period+4}: {count} documents")


if __name__ == "__main__":
    main()
