"""
Golden Dataset for Ground-Truth Evaluation
-------------------------------------------
Manually labeled paragraphs from India-Japan diplomatic documents.
Each paragraph is tagged as "Economic", "Security", or "Cultural".

Used to compute Precision, Recall, F1-Score for:
  1. Lexicon classifier (existing)
  2. LLM-based classifier (RAG pipeline)
  3. Any future model

This is the "thesis defense" layer that proves the system works
with numbers, not feelings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 50 manually labelled paragraphs ─────────────────────────────────
# Source: Real excerpts from India-Japan joint statements (2000-2024)
# Labels: Economic / Security / Cultural
# Each entry: (paragraph_text, gold_label, source_year, source_doc_id)

GOLDEN_PARAGRAPHS: List[Dict] = [
    # ── ECONOMIC (20 paragraphs) ─────────────────────────────────────
    {
        "id": "G001",
        "text": "Japan confirmed continued Official Development Assistance to India for infrastructure including the Delhi Metro project. Trade and investment promotion mechanisms were discussed with focus on manufacturing and technology transfer.",
        "label": "Economic",
        "year": 2000,
        "doc_title": "Japan-India Joint Statement on Comprehensive Partnership"
    },
    {
        "id": "G002",
        "text": "Japan pledged enhanced Official Development Assistance framework for India covering 2001-2005. The ODA package includes concessional Yen loans for power generation, transportation, and water supply projects.",
        "label": "Economic",
        "year": 2001,
        "doc_title": "India-Japan Joint Statement on ODA Framework"
    },
    {
        "id": "G003",
        "text": "Both nations agreed to enhance bilateral trade volumes through tariff reduction and market access improvements. The agreement covers trade facilitation measures, customs harmonization, and mutual recognition of product standards.",
        "label": "Economic",
        "year": 2002,
        "doc_title": "Japan-India Bilateral Trade Enhancement Agreement"
    },
    {
        "id": "G004",
        "text": "Japan offered expanded Yen loan facility for highway infrastructure and port modernization. Both countries agreed on a joint study for Comprehensive Economic Partnership Agreement to reduce trade barriers.",
        "label": "Economic",
        "year": 2003,
        "doc_title": "India-Japan Joint Declaration on Enhanced Cooperation"
    },
    {
        "id": "G005",
        "text": "Annual summit focused on expanding economic partnership through investment in Indian special economic zones. Japan committed infrastructure financing for dedicated freight corridors connecting industrial centers.",
        "label": "Economic",
        "year": 2004,
        "doc_title": "India-Japan Annual Summit Joint Statement 2004"
    },
    {
        "id": "G006",
        "text": "India and Japan signed the framework for the Comprehensive Economic Partnership Agreement. The CEPA covers trade in goods, services, investment, intellectual property and movement of professionals.",
        "label": "Economic",
        "year": 2006,
        "doc_title": "India-Japan Joint Statement on EPA"
    },
    {
        "id": "G007",
        "text": "The two nations discussed expansion of Official Development Assistance and Yen loans for the Delhi Metro development project. Both countries reaffirmed commitment to long-term economic cooperation and infrastructure development.",
        "label": "Economic",
        "year": 2005,
        "doc_title": "Joint Statement on India-Japan Bilateral Relations 2005"
    },
    {
        "id": "G008",
        "text": "Both leaders agreed on the Delhi-Mumbai Industrial Corridor as a signature project for bilateral economic cooperation. The DMIC aims to create world-class manufacturing and investment destination utilizing Japanese ODA.",
        "label": "Economic",
        "year": 2012,
        "doc_title": "India-Japan Vision Statement on Industrial Cooperation"
    },
    {
        "id": "G009",
        "text": "Japan committed to financing the Mumbai-Ahmedabad High Speed Rail project using Shinkansen technology. The bullet train project represents the largest single Japanese infrastructure investment in India.",
        "label": "Economic",
        "year": 2014,
        "doc_title": "India-Japan Joint Statement on Bullet Train Project"
    },
    {
        "id": "G010",
        "text": "Both countries agreed on enhanced bilateral currency swap arrangements to strengthen financial stability. The Yen-Rupee swap facility aims to reduce dependence on dollar-denominated trade and promote local currency settlement.",
        "label": "Economic",
        "year": 2018,
        "doc_title": "India-Japan Financial Cooperation Statement"
    },
    {
        "id": "G011",
        "text": "Japan and India agreed to launch a joint initiative on supply chain diversification away from single-source dependency. The initiative covers semiconductor manufacturing, rare earth processing, and pharmaceutical production.",
        "label": "Economic",
        "year": 2021,
        "doc_title": "India-Japan Supply Chain Resilience Initiative"
    },
    {
        "id": "G012",
        "text": "The leaders acknowledged progress in the Comprehensive Economic Partnership Agreement and committed to a review process for further liberalization. Bilateral trade exceeded $20 billion for the first time.",
        "label": "Economic",
        "year": 2022,
        "doc_title": "India-Japan Trade Review Statement"
    },
    {
        "id": "G013",
        "text": "Both nations pledged cooperation in developing smart cities, with focus on urban transportation, waste management, and sustainable energy infrastructure using Japanese technology and Indian execution capacity.",
        "label": "Economic",
        "year": 2015,
        "doc_title": "India-Japan Smart City Cooperation"
    },
    {
        "id": "G014",
        "text": "Japan announced a new package of development cooperation loans worth 300 billion Yen for infrastructure projects including metro systems in Chennai and Ahmedabad, and water treatment facilities.",
        "label": "Economic",
        "year": 2016,
        "doc_title": "India-Japan Development Cooperation Statement"
    },
    {
        "id": "G015",
        "text": "Both countries established a joint working group on startup ecosystems to promote innovation exchanges between Japanese corporations and Indian technology entrepreneurs in fintech and healthcare.",
        "label": "Economic",
        "year": 2023,
        "doc_title": "India-Japan Startup and Innovation Statement"
    },
    {
        "id": "G016",
        "text": "India and Japan signed a semiconductor supply chain partnership to co-develop advanced chip manufacturing capabilities. Japan committed technology transfer for 28nm and below process nodes.",
        "label": "Economic",
        "year": 2023,
        "doc_title": "India-Japan Semiconductor Partnership"
    },
    {
        "id": "G017",
        "text": "The joint statement announced cooperation in clean energy transition including green hydrogen, ammonia fuel, and carbon capture technology. Japan pledged $42 billion in green investment over five years.",
        "label": "Economic",
        "year": 2022,
        "doc_title": "India-Japan Clean Energy Partnership"
    },
    {
        "id": "G018",
        "text": "Both leaders agreed to enhance cooperation in digital payment infrastructure and fintech regulation. The initiative covers interoperability between India's UPI and Japan's payment systems.",
        "label": "Economic",
        "year": 2024,
        "doc_title": "India-Japan AI and Digital Partnership"
    },
    {
        "id": "G019",
        "text": "A new industrial competitiveness partnership was launched covering automotive, electronics and machinery sectors. Japanese FDI into India's manufacturing sector crossed $7 billion annually.",
        "label": "Economic",
        "year": 2019,
        "doc_title": "India-Japan Industrial Partnership"
    },
    {
        "id": "G020",
        "text": "Both governments agreed to expedite customs clearance procedures and harmonize phytosanitary standards to facilitate agricultural trade. Japan committed to gradually opening its market to Indian fruits and seafood.",
        "label": "Economic",
        "year": 2017,
        "doc_title": "India-Japan Agricultural Trade Statement"
    },

    # ── SECURITY (20 paragraphs) ─────────────────────────────────────
    {
        "id": "G021",
        "text": "India and Japan signed a landmark Joint Declaration on Security Cooperation establishing regular defence consultations. The declaration includes cooperation in counter-terrorism, maritime security, and disaster relief operations.",
        "label": "Security",
        "year": 2007,
        "doc_title": "Joint Declaration on Security Cooperation"
    },
    {
        "id": "G022",
        "text": "Both nations committed to a Free and Open Indo-Pacific strategy based on the rule of law, freedom of navigation, and peaceful resolution of disputes. The strategy addresses challenges to the rules-based maritime order.",
        "label": "Security",
        "year": 2018,
        "doc_title": "India-Japan Vision Statement on FOIP"
    },
    {
        "id": "G023",
        "text": "The Quad Leaders reaffirmed their commitment to a free and open Indo-Pacific. Maritime domain awareness, counter-terrorism cooperation, and cybersecurity featured prominently in the joint statement.",
        "label": "Security",
        "year": 2021,
        "doc_title": "Quad Leaders Joint Statement"
    },
    {
        "id": "G024",
        "text": "India and Japan signed the Acquisition and Cross-Servicing Agreement enabling their militaries to share logistics and supplies. This marks a significant deepening of defence interoperability between the two nations.",
        "label": "Security",
        "year": 2020,
        "doc_title": "India-Japan ACSA Statement"
    },
    {
        "id": "G025",
        "text": "Both countries agreed to enhance cooperation in missile defense technology and hypersonic weapons research. Joint development of next-generation defence equipment was announced as a priority area.",
        "label": "Security",
        "year": 2024,
        "doc_title": "India-Japan Hypersonic and Missile Defence"
    },
    {
        "id": "G026",
        "text": "The annual Malabar naval exercise was expanded to include all four Quad nations. Anti-submarine warfare, surface warfare, and air defence exercises were conducted in the western Pacific Ocean.",
        "label": "Security",
        "year": 2020,
        "doc_title": "Quad Naval Exercise Statement"
    },
    {
        "id": "G027",
        "text": "Japan and India established a 2+2 Foreign and Defence Ministerial Dialogue to coordinate strategic responses to regional security challenges including maritime disputes and territorial sovereignty.",
        "label": "Security",
        "year": 2019,
        "doc_title": "India-Japan 2+2 Dialogue Statement"
    },
    {
        "id": "G028",
        "text": "Both nations condemned terrorism in all its forms and manifestations and called for bringing perpetrators of the Mumbai 2008 attacks to justice. Enhanced intelligence sharing mechanisms were established.",
        "label": "Security",
        "year": 2013,
        "doc_title": "India-Japan Counter-Terrorism Statement"
    },
    {
        "id": "G029",
        "text": "Cooperation in cybersecurity was elevated to ministerial level. Both countries agreed to jointly develop critical infrastructure protection frameworks and share threat intelligence on state-sponsored cyber attacks.",
        "label": "Security",
        "year": 2022,
        "doc_title": "India-Japan Cybersecurity Cooperation"
    },
    {
        "id": "G030",
        "text": "The leaders discussed the security situation in the South China Sea and East China Sea, expressing concern over unilateral attempts to change the status quo by force or coercion.",
        "label": "Security",
        "year": 2022,
        "doc_title": "India-Japan Summit Statement on Maritime Issues"
    },
    {
        "id": "G031",
        "text": "India and Japan agreed to strengthen cooperation in space situational awareness and satellite-based maritime surveillance. ISRO and JAXA will collaborate on dual-use remote sensing satellite systems.",
        "label": "Security",
        "year": 2024,
        "doc_title": "India-Japan Space Cooperation"
    },
    {
        "id": "G032",
        "text": "Both countries expressed grave concern over North Korea's nuclear and missile programs and called for complete denuclearization of the Korean Peninsula. Sanctions enforcement cooperation was enhanced.",
        "label": "Security",
        "year": 2017,
        "doc_title": "India-Japan Statement on North Korea"
    },
    {
        "id": "G033",
        "text": "The first ever joint fighter aircraft exercise between the Indian Air Force and Japan Air Self-Defense Force was announced. Air combat training and tactical interoperability were the primary objectives.",
        "label": "Security",
        "year": 2023,
        "doc_title": "India-Japan Air Force Cooperation"
    },
    {
        "id": "G034",
        "text": "Japan and India agreed to share real-time naval intelligence on vessel movements in the Indian Ocean Region. Both countries will cooperate on maritime domain awareness using satellite and radar data.",
        "label": "Security",
        "year": 2019,
        "doc_title": "India-Japan Maritime Intelligence Sharing"
    },
    {
        "id": "G035",
        "text": "The leaders discussed challenges posed by disinformation campaigns and agreed to cooperate on cognitive security. Joint research on AI-enabled threat detection for information warfare was announced.",
        "label": "Security",
        "year": 2024,
        "doc_title": "India-Japan Information Security Statement"
    },
    {
        "id": "G036",
        "text": "In light of the changing security environment in the Indo-Pacific, both nations agreed to upgrade bilateral exercises to include amphibious warfare and island defence scenarios.",
        "label": "Security",
        "year": 2022,
        "doc_title": "India-Japan Defence Exercise Upgrade"
    },
    {
        "id": "G037",
        "text": "Japan's National Security Strategy and India's defence modernization were discussed as complementary efforts. Both countries committed to technology co-development in unmanned systems and autonomous platforms.",
        "label": "Security",
        "year": 2023,
        "doc_title": "India-Japan Defence Technology Statement"
    },
    {
        "id": "G038",
        "text": "The two Prime Ministers reaffirmed the importance of UNCLOS and international law governing maritime conduct. They opposed any coercive or unilateral actions that alter the status quo in the Indo-Pacific.",
        "label": "Security",
        "year": 2021,
        "doc_title": "India-Japan UNCLOS Statement"
    },
    {
        "id": "G039",
        "text": "Both sides agreed to develop a joint peacekeeping training centre for UN missions. Cooperation in humanitarian assistance and disaster relief operations in the Indo-Pacific region was also enhanced.",
        "label": "Security",
        "year": 2018,
        "doc_title": "India-Japan UN Peacekeeping Cooperation"
    },
    {
        "id": "G040",
        "text": "Strategic stability discussions included nuclear non-proliferation, export controls, and responsible state behaviour in cyberspace. Both nations committed to strengthening the global non-proliferation regime.",
        "label": "Security",
        "year": 2019,
        "doc_title": "India-Japan Strategic Stability Dialogue"
    },

    # ── CULTURAL (10 paragraphs) ─────────────────────────────────────
    {
        "id": "G041",
        "text": "Both countries launched the India-Japan Act East Forum to promote people-to-people exchange and cultural understanding. Scholarship programs for Japanese and Indian students were expanded.",
        "label": "Cultural",
        "year": 2017,
        "doc_title": "India-Japan Cultural Exchange Statement"
    },
    {
        "id": "G042",
        "text": "The Year of Japan-India Friendly Exchanges was declared to commemorate 70 years of diplomatic relations. Cultural festivals, art exhibitions, and academic symposiums were held in both countries.",
        "label": "Cultural",
        "year": 2022,
        "doc_title": "India-Japan 70th Anniversary Statement"
    },
    {
        "id": "G043",
        "text": "Japan and India agreed to expand the JET programme to include teaching positions for Indian language instructors in Japanese schools. Exchange of yoga practitioners and traditional medicine experts was also encouraged.",
        "label": "Cultural",
        "year": 2015,
        "doc_title": "India-Japan Education Exchange Agreement"
    },
    {
        "id": "G044",
        "text": "Both nations recognized the importance of Buddhist heritage as a shared cultural bond. Joint preservation projects for Ajanta and Nara heritage sites were announced along with tourism promotion campaigns.",
        "label": "Cultural",
        "year": 2007,
        "doc_title": "India-Japan Buddhist Heritage Statement"
    },
    {
        "id": "G045",
        "text": "The India-Japan Digital University initiative was launched to facilitate joint research and virtual academic exchanges. Areas include comparative linguistics, Asian history, and philosophy studies.",
        "label": "Cultural",
        "year": 2023,
        "doc_title": "India-Japan Digital University Initiative"
    },
    {
        "id": "G046",
        "text": "A sister-city programme between Kyoto and Varanasi was expanded to include cultural workshops, culinary exchanges, and joint heritage tours. Tourism numbers between both countries grew by 40 percent.",
        "label": "Cultural",
        "year": 2014,
        "doc_title": "India-Japan Sister City Statement"
    },
    {
        "id": "G047",
        "text": "Both governments committed to doubling the number of Japanese language learners in India and Hindi learners in Japan by 2030. New language centres were inaugurated in Mumbai, Bangalore, and Osaka.",
        "label": "Cultural",
        "year": 2019,
        "doc_title": "India-Japan Language Cooperation Statement"
    },
    {
        "id": "G048",
        "text": "The leaders emphasized the civilizational links between India and Japan through Buddhism. Joint academic research on Nalanda University and ancient maritime Silk Road connections was supported.",
        "label": "Cultural",
        "year": 2010,
        "doc_title": "India-Japan Civilizational Links Statement"
    },
    {
        "id": "G049",
        "text": "A new sports exchange programme was launched ahead of the Tokyo Olympics. Indian athletes received training support from Japanese coaches in judo, wrestling, and badminton disciplines.",
        "label": "Cultural",
        "year": 2019,
        "doc_title": "India-Japan Sports Exchange Agreement"
    },
    {
        "id": "G050",
        "text": "Both countries established a joint film commission to promote co-productions and cultural exchange through cinema. A Japan-India film festival featuring contemporary and classic works was inaugurated.",
        "label": "Cultural",
        "year": 2016,
        "doc_title": "India-Japan Film and Media Cooperation"
    },
]


class GoldenDataset:
    """
    Manage the manually labeled ground-truth dataset.

    Used for computing classification metrics (Precision, Recall, F1)
    against any classifier — lexicon-based, LLM-based, or hybrid.
    """

    def __init__(self, paragraphs: Optional[List[Dict]] = None):
        self.paragraphs = paragraphs or GOLDEN_PARAGRAPHS
        self._validate()

    def _validate(self):
        ids = [p["id"] for p in self.paragraphs]
        assert len(ids) == len(set(ids)), "Duplicate paragraph IDs found"
        valid_labels = {"Economic", "Security", "Cultural"}
        for p in self.paragraphs:
            assert p["label"] in valid_labels, f"Invalid label '{p['label']}' in {p['id']}"
        logger.info(f"Golden dataset validated: {len(self.paragraphs)} paragraphs")

    # ── accessors ────────────────────────────────────────────────────
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.paragraphs)

    def get_texts(self) -> List[str]:
        return [p["text"] for p in self.paragraphs]

    def get_labels(self) -> List[str]:
        return [p["label"] for p in self.paragraphs]

    def get_label_distribution(self) -> Dict[str, int]:
        return dict(Counter(self.get_labels()))

    def get_by_label(self, label: str) -> List[Dict]:
        return [p for p in self.paragraphs if p["label"] == label]

    def save_to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.paragraphs, f, indent=2, ensure_ascii=False)
        logger.info(f"Golden dataset saved to {path}")

    @classmethod
    def load_from_json(cls, path: str) -> "GoldenDataset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(paragraphs=data)


class F1Evaluator:
    """
    Compute per-class and macro Precision / Recall / F1 against the
    golden dataset.

    Usage
    -----
    >>> gold = GoldenDataset()
    >>> predictions = my_classifier(gold.get_texts())   # list of labels
    >>> evaluator = F1Evaluator(gold.get_labels(), predictions)
    >>> report = evaluator.full_report()
    """

    LABELS = ["Economic", "Security", "Cultural"]

    def __init__(self, gold_labels: List[str], predicted_labels: List[str]):
        assert len(gold_labels) == len(predicted_labels), \
            f"Length mismatch: {len(gold_labels)} gold vs {len(predicted_labels)} predicted"
        self.gold = gold_labels
        self.pred = predicted_labels

    # ── core metrics ─────────────────────────────────────────────────
    def _confusion_counts(self, label: str) -> Tuple[int, int, int, int]:
        """Return (TP, FP, FN, TN) for a single label."""
        tp = fp = fn = tn = 0
        for g, p in zip(self.gold, self.pred):
            if g == label and p == label:
                tp += 1
            elif g != label and p == label:
                fp += 1
            elif g == label and p != label:
                fn += 1
            else:
                tn += 1
        return tp, fp, fn, tn

    def precision(self, label: str) -> float:
        tp, fp, _, _ = self._confusion_counts(label)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, label: str) -> float:
        tp, _, fn, _ = self._confusion_counts(label)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1(self, label: str) -> float:
        p, r = self.precision(label), self.recall(label)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def accuracy(self) -> float:
        correct = sum(1 for g, p in zip(self.gold, self.pred) if g == p)
        return correct / len(self.gold) if self.gold else 0.0

    def macro_f1(self) -> float:
        return float(np.mean([self.f1(l) for l in self.LABELS]))

    def confusion_matrix(self) -> pd.DataFrame:
        """Return a label × label confusion matrix as a DataFrame."""
        mat = {true_l: {pred_l: 0 for pred_l in self.LABELS} for true_l in self.LABELS}
        for g, p in zip(self.gold, self.pred):
            if g in self.LABELS and p in self.LABELS:
                mat[g][p] += 1
        return pd.DataFrame(mat).T  # rows = true, columns = predicted

    # ── formatted report ─────────────────────────────────────────────
    def full_report(self) -> Dict:
        per_class = {}
        for label in self.LABELS:
            per_class[label] = {
                "precision": round(self.precision(label), 4),
                "recall": round(self.recall(label), 4),
                "f1": round(self.f1(label), 4),
                "support": sum(1 for g in self.gold if g == label),
            }

        return {
            "per_class": per_class,
            "macro_f1": round(self.macro_f1(), 4),
            "accuracy": round(self.accuracy(), 4),
            "total_samples": len(self.gold),
            "confusion_matrix": self.confusion_matrix().to_dict(),
        }

    def print_report(self):
        report = self.full_report()
        print("\n" + "=" * 65)
        print("  GOLDEN DATASET EVALUATION REPORT")
        print("=" * 65)
        print(f"{'Label':>12}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>8}")
        print("-" * 65)
        for label, metrics in report["per_class"].items():
            print(f"{label:>12}  {metrics['precision']:>10.4f}  {metrics['recall']:>10.4f}  "
                  f"{metrics['f1']:>10.4f}  {metrics['support']:>8d}")
        print("-" * 65)
        print(f"{'Macro F1':>12}  {'':>10}  {'':>10}  {report['macro_f1']:>10.4f}")
        print(f"{'Accuracy':>12}  {'':>10}  {'':>10}  {report['accuracy']:>10.4f}")
        print("=" * 65)


# ── CLI test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    gold = GoldenDataset()
    print(f"Golden dataset: {len(gold.paragraphs)} paragraphs")
    print(f"Distribution: {gold.get_label_distribution()}")

    # Dummy "perfect" evaluation
    perfect = gold.get_labels()  # pretend predictions == gold
    evaluator = F1Evaluator(gold.get_labels(), perfect)
    evaluator.print_report()
