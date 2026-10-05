"""
src/rag/knowledge_base.py
==========================
Medical knowledge base: document loading, chunking, metadata management.

Documents are stored as JSON with the schema:
{
  "title": "...",
  "source": "...",
  "url": "...",
  "date": "...",
  "text": "...",
  "category": "..."
}
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/rag.yaml")
DOCUMENTS_DIR = Path("knowledge_base/documents")
PROCESSED_DIR = Path("knowledge_base/processed")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    title: str
    source: str
    url: str
    date: str
    text: str
    category: str
    chunk_id: int = 0
    total_chunks: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class MedicalTextChunker:
    """
    Sentence-aware chunker that avoids breaking medical statements mid-sentence.
    """

    def __init__(self, chunk_size: int = 600, overlap: int = 75, min_chunk: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk = min_chunk

    def _split_sentences(self, text: str) -> list[str]:
        """Split text on sentence boundaries."""
        # Simple sentence splitter — handles common medical abbreviations
        text = re.sub(r"\s+", " ", text).strip()
        # Protect common abbreviations
        protected = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Fig|vs|etc|approx|e\.g|i\.e)\.", r"\1<DOT>", text)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
        return [s.replace("<DOT>", ".") for s in sentences if s.strip()]

    def chunk(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks of approximately chunk_size characters.
        Respects sentence boundaries.
        """
        sentences = self._split_sentences(text)
        chunks: list[str] = []
        current_chars = 0
        current_sentences: list[str] = []

        for sent in sentences:
            sent_len = len(sent)

            if current_chars + sent_len > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                if len(chunk_text) >= self.min_chunk:
                    chunks.append(chunk_text)

                # Overlap: carry last sentences forward
                overlap_chars = 0
                carry: list[str] = []
                for s in reversed(current_sentences):
                    if overlap_chars + len(s) <= self.overlap:
                        carry.insert(0, s)
                        overlap_chars += len(s)
                    else:
                        break
                current_sentences = carry
                current_chars = sum(len(s) for s in carry)

            current_sentences.append(sent)
            current_chars += sent_len

        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if len(chunk_text) >= self.min_chunk:
                chunks.append(chunk_text)

        return chunks


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

class MedicalKnowledgeBase:
    """
    Manages the medical knowledge base: loading, chunking, and serializing documents.
    """

    def __init__(
        self,
        documents_dir: Path = DOCUMENTS_DIR,
        processed_dir: Path = PROCESSED_DIR,
        config_path: Path = CONFIG_PATH,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.processed_dir = Path(processed_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        cfg = self._load_config(config_path)
        chunking = cfg.get("chunking", {})
        self.chunker = MedicalTextChunker(
            chunk_size=chunking.get("chunk_size", 600),
            overlap=chunking.get("chunk_overlap", 75),
            min_chunk=chunking.get("min_chunk_size", 100),
        )

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            with open(path) as fh:
                return yaml.safe_load(fh) or {}
        return {}

    # ------------------------------------------------------------------
    def load_json_documents(self) -> list[Document]:
        """Load all .json documents from the documents directory."""
        docs: list[Document] = []
        for path in self.documents_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, list):
                    for d in raw:
                        docs.append(Document.from_dict(d))
                else:
                    docs.append(Document.from_dict(raw))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)
        logger.info("Loaded %d documents from %s", len(docs), self.documents_dir)
        return docs

    # ------------------------------------------------------------------
    def chunk_documents(self, docs: list[Document]) -> list[Document]:
        """Split documents into chunks and return a flat list of chunk Documents."""
        chunks: list[Document] = []
        for doc in docs:
            if not doc.text.strip():
                continue
            text_chunks = self.chunker.chunk(doc.text)
            for i, chunk_text in enumerate(text_chunks):
                chunk_doc = Document(
                    title=doc.title,
                    source=doc.source,
                    url=doc.url,
                    date=doc.date,
                    text=chunk_text,
                    category=doc.category,
                    chunk_id=i,
                    total_chunks=len(text_chunks),
                )
                chunks.append(chunk_doc)
        logger.info(
            "Chunked %d documents → %d chunks", len(docs), len(chunks)
        )
        return chunks

    # ------------------------------------------------------------------
    def save_chunks(self, chunks: list[Document]) -> Path:
        """Save chunked documents to the processed directory."""
        out_path = self.processed_dir / "chunks.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in chunks], fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d chunks to %s", len(chunks), out_path)
        return out_path

    # ------------------------------------------------------------------
    def load_chunks(self) -> list[Document]:
        """Load saved chunks from processed directory."""
        out_path = self.processed_dir / "chunks.json"
        if not out_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found at {out_path}. "
                "Run python -m src.rag.build_index first."
            )
        with open(out_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        chunks = [Document.from_dict(d) for d in raw]
        logger.info("Loaded %d chunks from %s", len(chunks), out_path)
        return chunks

    # ------------------------------------------------------------------
    def add_synthetic_medical_documents(self) -> list[Document]:
        """
        Add a curated set of foundational medical knowledge documents.
        These are factual, educational summaries derived from public health information.
        They are NOT AI-generated fiction — they are structured summaries of
        well-established medical knowledge.
        """
        synthetic_docs = _FOUNDATIONAL_MEDICAL_DOCS
        logger.info("Adding %d foundational medical documents.", len(synthetic_docs))
        out_paths: list[Document] = []
        for doc_dict in synthetic_docs:
            doc = Document.from_dict(doc_dict)
            # Save to documents directory for persistence
            safe_title = re.sub(r"[^\w\s-]", "", doc.title).replace(" ", "_")[:60]
            out_file = self.documents_dir / f"{safe_title}.json"
            if not out_file.exists():
                with open(out_file, "w", encoding="utf-8") as fh:
                    json.dump(doc.to_dict(), fh, indent=2)
            out_paths.append(doc)
        return out_paths


# ---------------------------------------------------------------------------
# Foundational medical knowledge documents
# (Public health information — educational / non-diagnostic)
# ---------------------------------------------------------------------------

_FOUNDATIONAL_MEDICAL_DOCS: list[dict] = [
    {
        "title": "Common Cold: Symptoms, Causes, and Treatment",
        "source": "MedlinePlus / NIH",
        "url": "https://medlineplus.gov/commoncold.html",
        "date": "2024",
        "category": "respiratory",
        "text": textwrap.dedent("""\
            The common cold is a viral infection of the upper respiratory tract.
            It is caused by over 200 different viruses, most commonly rhinoviruses.

            Symptoms typically include runny or stuffy nose, sore throat, cough, sneezing,
            mild headache, mild body aches, and low-grade fever (more common in children).
            Symptoms usually appear 1-3 days after exposure and last 7-10 days.

            Treatment is supportive. Rest, fluids, and over-the-counter medications may
            relieve symptoms. Antibiotics are NOT effective against viral infections.

            See a doctor if symptoms are severe, last more than 10 days, or include
            high fever, severe headache, chest pain, or difficulty breathing.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Influenza (Flu): Symptoms, Complications, and Prevention",
        "source": "CDC",
        "url": "https://www.cdc.gov/flu/symptoms/index.html",
        "date": "2024",
        "category": "respiratory",
        "text": textwrap.dedent("""\
            Influenza (flu) is a contagious respiratory illness caused by influenza viruses.
            It can cause mild to severe illness and can lead to hospitalisation or death.

            Symptoms of flu include sudden onset of fever (usually 100–104°F / 38–40°C),
            chills, cough, sore throat, runny or stuffy nose, muscle or body aches,
            headaches, and fatigue. Some people have vomiting and diarrhea (more common in children).

            Key difference from cold: flu symptoms are typically more severe and appear suddenly.
            Colds develop gradually.

            Complications include pneumonia, bronchitis, sinus infections, ear infections,
            and worsening of chronic conditions.

            Prevention: Annual flu vaccination is the best protection. Handwashing and
            avoiding close contact with sick individuals also helps.

            Antiviral medications (e.g., oseltamivir/Tamiflu) can reduce severity if
            started within 48 hours of symptom onset. These require a prescription.

            Seek medical care promptly for: difficulty breathing, persistent chest pain,
            confusion, severe vomiting, or symptoms that improve then return with fever/cough.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Hypertension (High Blood Pressure): Overview",
        "source": "American Heart Association / CDC",
        "url": "https://www.cdc.gov/bloodpressure/index.html",
        "date": "2024",
        "category": "cardiovascular",
        "text": textwrap.dedent("""\
            High blood pressure (hypertension) is when blood pressure consistently reads
            130/80 mmHg or higher. It is called the "silent killer" because it often has
            no symptoms.

            Normal blood pressure: below 120/80 mmHg.
            Elevated: 120-129 systolic and less than 80 diastolic.
            Stage 1 hypertension: 130-139/80-89 mmHg.
            Stage 2 hypertension: 140/90 mmHg or higher.
            Hypertensive crisis: above 180/120 mmHg — seek emergency care.

            Causes: include genetics, obesity, high sodium diet, physical inactivity,
            excessive alcohol, chronic kidney disease, and certain medications.

            Long-term complications include heart attack, stroke, kidney disease,
            heart failure, and vision problems.

            Lifestyle management: reducing sodium intake, regular physical activity,
            maintaining healthy weight, limiting alcohol, not smoking.

            Many people require medication. Never stop blood pressure medication without
            consulting a doctor. Multiple medication classes exist (ACE inhibitors,
            ARBs, calcium channel blockers, diuretics, etc.) — selection depends on
            individual patient factors.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Type 2 Diabetes: Symptoms, Causes, and Management",
        "source": "CDC / NIH",
        "url": "https://www.cdc.gov/diabetes/basics/type2.html",
        "date": "2024",
        "category": "endocrine",
        "text": textwrap.dedent("""\
            Type 2 diabetes is a condition where the body does not use insulin properly
            (insulin resistance) and eventually cannot make enough insulin.

            Risk factors: overweight, physical inactivity, family history, age over 45,
            prediabetes, gestational diabetes history, certain ethnicities.

            Symptoms: increased thirst and urination, unexplained weight loss, blurred vision,
            slow-healing wounds, frequent infections, numbness or tingling in hands/feet,
            fatigue. Many people have no symptoms initially.

            Diagnosis: fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or 2-hour glucose ≥200 mg/dL.

            Management: lifestyle changes (diet, exercise, weight management) are first-line.
            Medications include metformin, SGLT2 inhibitors, GLP-1 agonists, and others.
            Some people with type 2 may require insulin over time.

            Complications: if uncontrolled — heart disease, stroke, kidney disease, neuropathy,
            retinopathy, foot complications.

            Type 1 diabetes is different: it is autoimmune, usually diagnosed in younger people,
            and always requires insulin. It is not caused by lifestyle.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Iron Deficiency Anemia: Symptoms and Treatment",
        "source": "MedlinePlus / NIH",
        "url": "https://medlineplus.gov/irondeficiencyanemia.html",
        "date": "2024",
        "category": "hematology",
        "text": textwrap.dedent("""\
            Iron deficiency anemia is the most common form of anemia worldwide. It occurs
            when the body does not have enough iron to produce hemoglobin.

            Common causes: inadequate dietary iron, poor absorption, blood loss (menstruation,
            gastrointestinal bleeding), or increased iron requirements (pregnancy, growth).

            Symptoms: fatigue, weakness, pale skin, cold hands/feet, brittle nails, dizziness,
            shortness of breath on exertion, headache, difficulty concentrating, rapid heartbeat,
            unusual cravings for non-nutritive substances (pica — ice, dirt).

            Diagnosis: complete blood count (CBC), serum ferritin, serum iron, transferrin saturation.

            Treatment: dietary changes (red meat, leafy greens, legumes, fortified foods),
            vitamin C to enhance iron absorption, iron supplements (often cause constipation;
            take with food if needed), and treatment of underlying cause.

            Never self-diagnose and treat anemia. Blood tests are required for diagnosis.
            Unexplained anemia should be evaluated by a doctor to rule out serious causes.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Dehydration: Causes, Symptoms, and Treatment",
        "source": "MedlinePlus / NIH",
        "url": "https://medlineplus.gov/dehydration.html",
        "date": "2024",
        "category": "general",
        "text": textwrap.dedent("""\
            Dehydration occurs when the body loses more fluids than it takes in.
            Even mild dehydration can affect physical and mental performance.

            Common causes: inadequate fluid intake, excessive sweating, diarrhea, vomiting,
            fever, frequent urination, burns, or hot weather.

            Mild to moderate symptoms: thirst, dry mouth, dark yellow urine, decreased urine output,
            headache, dizziness, fatigue, dry skin, muscle cramps.

            Severe symptoms: very dark or no urine, rapid heartbeat, rapid breathing, sunken eyes,
            no tears when crying, confusion, fainting.

            Treatment for mild dehydration: oral rehydration with water or electrolyte solutions.
            Avoid caffeinated or alcoholic drinks.

            Seek emergency care for: severe confusion, no urine output for 8+ hours, fainting,
            severe diarrhea/vomiting that prevents fluid intake, suspected heatstroke.

            Daily fluid needs vary: approximately 8 cups (2 litres) for adults in normal conditions,
            more with exercise, heat, illness, or pregnancy.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Ibuprofen: Uses, Side Effects, and Safety",
        "source": "MedlinePlus / NIH",
        "url": "https://medlineplus.gov/druginfo/meds/a682159.html",
        "date": "2024",
        "category": "pharmacology",
        "text": textwrap.dedent("""\
            Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to relieve pain,
            reduce fever, and decrease inflammation.

            Common uses: headaches, toothaches, menstrual cramps, muscle aches, minor arthritis,
            back pain, and fever.

            Common side effects: stomach upset, nausea, heartburn, dizziness, headache.
            Taking with food or milk reduces stomach upset.

            Serious risks: gastrointestinal bleeding or ulcers (especially with long-term use),
            kidney damage (especially with dehydration or pre-existing kidney disease),
            increased cardiovascular risk (heart attack, stroke) with long-term use or high doses,
            allergic reactions.

            Do not take ibuprofen if you have: active stomach ulcers, severe kidney or liver disease,
            aspirin-sensitive asthma, or are in the last trimester of pregnancy without medical advice.

            Drug interactions: blood thinners (warfarin), other NSAIDs, certain blood pressure
            medications, lithium, and others. Always check with a pharmacist or doctor.

            Standard adult dose information is on the package; do not exceed recommended doses.
            Always read the label and consult a pharmacist or doctor if uncertain.

            THIS IS GENERAL INFORMATION ONLY. Dosing decisions should be made with a healthcare provider.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Chest Pain: When Is It an Emergency?",
        "source": "American Heart Association",
        "url": "https://www.heart.org/en/health-topics/heart-attack/warning-signs-of-a-heart-attack",
        "date": "2024",
        "category": "emergency",
        "text": textwrap.dedent("""\
            Chest pain must always be taken seriously. While many causes are benign
            (muscle strain, acid reflux, anxiety), chest pain can indicate a life-threatening
            condition.

            CALL EMERGENCY SERVICES IMMEDIATELY (911) for:
            - Chest pain with pressure, squeezing, tightness, or fullness
            - Pain radiating to arm, jaw, neck, shoulder, or back
            - Chest pain with shortness of breath
            - Chest pain with sweating, nausea, or vomiting
            - Sudden severe chest pain at rest
            - Chest pain with fainting or near-fainting
            - Chest pain lasting more than a few minutes or coming and going

            These may indicate heart attack, pulmonary embolism (blood clot in lungs),
            aortic dissection, or other emergencies.

            Do not drive yourself to the hospital. Call emergency services.

            Less urgent causes include: acid reflux / heartburn, costochondritis (chest wall
            inflammation), pleuritis, anxiety/panic attacks, or musculoskeletal pain.
            Even if you suspect a benign cause, seek medical evaluation for new or unusual chest pain.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Sleep Hygiene: Tips for Better Sleep",
        "source": "CDC / NIH",
        "url": "https://www.cdc.gov/sleep/about_sleep/sleep_hygiene.html",
        "date": "2024",
        "category": "wellness",
        "text": textwrap.dedent("""\
            Good sleep is essential for physical and mental health. Adults need 7-9 hours
            per night. Chronic poor sleep increases risk of obesity, diabetes, cardiovascular
            disease, and mental health problems.

            Evidence-based sleep hygiene recommendations:
            - Maintain a consistent sleep schedule (same bedtime and wake time, even weekends)
            - Keep bedroom cool, dark, and quiet
            - Avoid screens (phones, tablets, TVs) 30-60 minutes before bed
            - Avoid caffeine after early afternoon (caffeine half-life ~5-6 hours)
            - Avoid alcohol close to bedtime (disrupts sleep architecture)
            - Regular physical activity helps, but not vigorous exercise within 2-3 hours of sleep
            - Avoid large meals close to bedtime
            - Develop a relaxing pre-sleep routine

            Common sleep disorders include insomnia, obstructive sleep apnea, restless legs
            syndrome, and narcolepsy. These require medical evaluation.

            See a doctor if: sleep problems persist for weeks, you frequently wake unrefreshed,
            you snore loudly or stop breathing during sleep, or daytime sleepiness affects safety.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Fever: Causes, Management, and When to See a Doctor",
        "source": "Mayo Clinic / CDC",
        "url": "https://www.mayoclinic.org/diseases-conditions/fever/symptoms-causes/syc-20352759",
        "date": "2024",
        "category": "general",
        "text": textwrap.dedent("""\
            A fever is a temporary rise in body temperature, usually in response to illness.
            Normal body temperature is approximately 37°C (98.6°F), though this varies by person.
            A fever is generally defined as a temperature above 38°C (100.4°F).

            Common causes: viral infections (colds, flu, COVID-19), bacterial infections,
            heat exhaustion, certain medications, inflammatory conditions.

            Mild fever management: rest, adequate fluid intake, over-the-counter antipyretics
            (paracetamol/acetaminophen or ibuprofen) if needed to relieve discomfort.
            Fever is part of the immune response — not always necessary to suppress.

            Seek prompt medical care for:
            - Fever above 39.4°C (103°F) in adults
            - Fever with severe headache, stiff neck, confusion, or rash (possible meningitis)
            - Fever with difficulty breathing, chest pain, or severe abdominal pain
            - Fever lasting more than 3 days
            - Any fever in a baby under 3 months old — seek emergency care
            - Fever with recent travel to areas with tropical disease risk

            Do not give aspirin to children or teenagers with fever (risk of Reye's syndrome).
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Persistent Cough: Causes and When to See a Doctor",
        "source": "MedlinePlus / NIH",
        "url": "https://medlineplus.gov/cough.html",
        "date": "2024",
        "category": "respiratory",
        "text": textwrap.dedent("""\
            A cough lasting more than 3 weeks is considered a persistent or chronic cough.
            Persistent cough is one of the most common reasons for medical visits.

            Common causes of persistent cough:
            - Postnasal drip from allergies or sinus problems
            - Gastroesophageal reflux disease (GERD)
            - Asthma (often presents as cough, especially at night)
            - ACE inhibitor medication side effect (very common)
            - Upper respiratory infection aftermath (post-viral cough)
            - Chronic obstructive pulmonary disease (COPD) — especially in smokers
            - Less commonly: pertussis (whooping cough), tuberculosis, lung cancer

            Red flags requiring urgent medical evaluation:
            - Coughing up blood (haemoptysis) — always urgent
            - Unintentional weight loss
            - Night sweats with cough
            - Persistent cough with fever and shortness of breath
            - Cough in a long-term smoker that changes character
            - Cough in immunocompromised individuals

            See a doctor if cough persists more than 3 weeks without obvious cause,
            or sooner if accompanied by concerning symptoms.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Drug Interactions: General Principles",
        "source": "FDA / MedlinePlus",
        "url": "https://www.fda.gov/consumers/consumer-updates/understanding-over-counter-medicines",
        "date": "2024",
        "category": "pharmacology",
        "text": textwrap.dedent("""\
            Drug interactions occur when one medication affects how another works,
            or when food/substances interact with medications.

            Types of drug interactions:
            - Drug-drug interactions: two or more medications interact
            - Drug-food interactions: food alters drug absorption or metabolism
            - Drug-condition interactions: a medical condition makes a drug harmful

            Important examples:
            - Warfarin interacts with many drugs and foods (especially vitamin K-rich foods)
            - St John's Wort reduces effectiveness of many medications
            - Grapefruit juice inhibits enzymes that metabolise many drugs
            - MAOIs interact dangerously with many drugs and foods

            Always tell your doctor and pharmacist about ALL medications, supplements,
            vitamins, and herbal products you take.

            THIS AI CANNOT PROVIDE ACCURATE PERSONALISED DRUG INTERACTION CHECKS.
            For drug interaction verification, always consult a pharmacist or doctor,
            or use an authoritative drug interaction checker tool.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
    {
        "title": "Mental Health: Anxiety and Depression Overview",
        "source": "NIMH / WHO",
        "url": "https://www.nimh.nih.gov/health/topics/anxiety-disorders",
        "date": "2024",
        "category": "mental_health",
        "text": textwrap.dedent("""\
            Anxiety disorders and depression are among the most common mental health conditions.

            Anxiety disorders include generalised anxiety disorder (GAD), panic disorder,
            social anxiety disorder, specific phobias, and others.
            Common symptoms: persistent worry, restlessness, fatigue, difficulty concentrating,
            irritability, muscle tension, sleep problems, panic attacks.

            Depression (major depressive disorder) symptoms: persistent low mood, loss of interest
            or pleasure, changes in appetite/weight, sleep disturbances, fatigue, feelings of
            worthlessness, difficulty concentrating, thoughts of death or suicide.

            Both conditions are treatable. Treatment options include psychotherapy
            (cognitive behavioural therapy is well-evidenced), medications (antidepressants,
            anti-anxiety medications), lifestyle changes, and combinations.

            IMPORTANT: If you are having thoughts of suicide or self-harm, please seek
            immediate help. Contact a crisis line or emergency services.

            Seeking professional mental health support is a sign of strength.
            A doctor, psychiatrist, or psychologist can help find the right treatment.
        """),
        "chunk_id": 0, "total_chunks": 1,
    },
]
