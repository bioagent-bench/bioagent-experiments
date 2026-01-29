import gzip
from pathlib import Path
import csv, math, random, string

def generate_decoy_mouse(path="DEA_PSV24.csv", n=20000, seed=42, all_zero_prob=0.35, missing_pq_prob=0.10):
    rng = random.Random(seed)

    header = [
        "", "gene_id", "gene_name", "feature", "fc", "log2fc", "pval", "qval",
        "wt.MeanFPKM", "tg.MeanFPKM",
        "FPKM.tau_02", "FPKM.tau_03", "FPKM.tau_04", "FPKM.tau_01", "FPKM.tau_05", "FPKM.tau_06"
    ]

    def gene_id(i):
        return f"ENSMUSG{(i % 10**11):011d}"

    def gene_name():
        L = rng.randint(3, 8)
        return rng.choice(string.ascii_uppercase) + "".join(
            rng.choice(string.ascii_lowercase + string.digits) for _ in range(L - 1)
        )

    def fmt(x, dec=6):
        s = f"{x:.{dec}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i in range(1, n + 1):
            gid = gene_id(i)
            gnm = gene_name()

            if rng.random() < all_zero_prob:
                w.writerow([
                    i, gid, gnm, "gene",
                    "1.0", "0.0", "", "",
                    "0.0", "0.0",
                    "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"
                ])
                continue

            tau = []
            for _ in range(6):
                mu = rng.uniform(-0.2, 2.3)
                sigma = rng.uniform(0.2, 0.8)
                val = math.exp(rng.gauss(mu, sigma))
                if rng.random() < 0.05:
                    val *= rng.uniform(0.0, 0.2)
                tau.append(val)

            wt_mean = sum(tau) / 6.0
            fc_true = math.exp(rng.gauss(0.0, 0.35))  
            tg_mean = wt_mean * fc_true

            wt_mean *= rng.uniform(0.95, 1.05)
            tg_mean *= rng.uniform(0.95, 1.05)

            wt_mean = max(wt_mean, 1e-12)
            tg_mean = max(tg_mean, 1e-12)

            fc = tg_mean / wt_mean
            log2fc = math.log(fc, 2)

            if rng.random() < missing_pq_prob:
                pval, qval = "", ""
            else:
                p = min(max(rng.random() ** 0.35, 0.0), 1.0)
                q = min(max(p * rng.uniform(1.0, 2.5), 0.0), 1.0)
                pval, qval = fmt(p, 12), fmt(q, 12)

            tau_01, tau_02, tau_03, tau_04, tau_05, tau_06 = tau

            w.writerow([
                i, gid, gnm, "gene",
                fmt(fc, 12), fmt(log2fc, 12),
                pval, qval,
                fmt(wt_mean, 12), fmt(tg_mean, 12),
                fmt(tau_02, 6), fmt(tau_03, 6), fmt(tau_04, 6),
                fmt(tau_01, 6), fmt(tau_05, 6), fmt(tau_06, 6)
            ])


def generate_decoy_fibrosis() -> str:
    import random
    from datetime import datetime

    rng = random.Random() 

    chrom = "7"
    start_pos, end_pos = 16000, 61000
    n_variants = 250
    n_samples = 17
    samples = [f"FA{i:05d}" for i in range(1, n_samples + 1)]
    bases = ["A", "C", "G", "T"]

    def rand_ref_alts():
        """Return (REF, [ALT...]) with SNPs + small indels + occasional multiallelic."""
        ref = rng.choice(bases)
        alts = []

        if rng.random() < 0.18:

            if rng.random() < 0.5:

                ins = "".join(rng.choice(bases) for _ in range(rng.randint(1, 4)))
                alts.append(ref + ins)
            else:
                extra = "".join(rng.choice(bases) for _ in range(rng.randint(1, 6)))
                ref = ref + extra
                alts.append(ref[0])
        else:
            alt = rng.choice([b for b in bases if b != ref])
            alts.append(alt)

        if len(ref) == 1 and rng.random() < 0.06:
            alt2 = rng.choice([b for b in bases if b != ref and b != alts[0]])
            alts.append(alt2)

        return ref, alts

    def format_gt(a, b):
        sep = "|" if rng.random() < 0.72 else "/"
        if rng.random() < 0.02:
            return f"{a}{sep}."
        if rng.random() < 0.02:
            return f".{sep}{b}"
        return f"{a}{sep}{b}"

    def sample_gt(num_alts):
        r = rng.random()
        if r < 0.08:
            return "./."
        if r < 0.12:
            return "."
        if num_alts == 1:
            weights = [
                ((0, 0), 0.52),
                ((0, 1), 0.28),
                ((1, 1), 0.12),
                ((1, 0), 0.08),
            ]
        else:
            weights = [
                ((0, 0), 0.46),
                ((0, 1), 0.22),
                ((1, 1), 0.10),
                ((0, 2), 0.10),
                ((2, 2), 0.04),
                ((1, 2), 0.04),
                ((2, 1), 0.04),
            ]
        x = rng.random()
        cum = 0.0
        for (a, b), w in weights:
            cum += w
            if x <= cum:
                return format_gt(a, b)
        a, b = weights[-1][0]
        return format_gt(a, b)

    def compute_ac_an(gt_strs, num_alts):
        ac = [0] * num_alts
        an = 0
        for gt in gt_strs:
            if gt in ("./.", ".|."):
                continue
            sep = "/" if "/" in gt else "|"
            parts = gt.split(sep)
            for p in parts:
                if p == ".":
                    continue
                try:
                    ai = int(p)
                except ValueError:
                    continue
                if ai == 0:
                    an += 1
                elif 1 <= ai <= num_alts:
                    an += 1
                    ac[ai - 1] += 1
        return ac, an

    def make_ann(pos, ref, alts):
        gene_name = "CHR_START-AC093627.7"
        gene_id = "CHR_START-ENSG00000232325"

        entries = []
        for alt in alts:
            # rough HGVS-ish (decoy)
            if len(ref) == 1 and len(alt) == 1:
                hgvs = f"n.{pos}{ref}>{alt}"
            elif len(ref) < len(alt):
                hgvs = f"n.{pos}_{pos+len(ref)-1}ins{alt[len(ref):]}"
            elif len(ref) > len(alt):
                del_seq = ref[len(alt):]
                hgvs = f"n.{pos+len(alt)}_{pos+len(ref)-1}del{del_seq}"
            else:
                hgvs = f"n.{pos}{ref}>{alt}"

            entries.append(
                f"{alt}|intergenic_region|MODIFIER|{gene_name}|{gene_id}|intergenic_region|{gene_id}|||{hgvs}||||||"
            )
        return ",".join(entries)

    # --- header (modeled after your sample) ---
    file_date = datetime.utcnow().strftime("%Y%m%d")
    header = []
    header.append("##fileformat=VCFv4.1\n")
    header.append(f"##fileDate={file_date}\n")
    header.append("##center=Complete Genomics\n")
    header.append("##source=CGAPipeline_2.0.0.26;cgatools_1.6.0\n")
    header.append("##source_GENOME_REFERENCE=NCBI build 37\n")
    header.append("##phasing=partial\n")
    header.append('##ALT=<ID=CGA_NOCALL,Description="No-called record">\n')
    header.append('##ALT=<ID=CGA_CNVWIN,Description="Copy number analysis window">\n')
    header.append('##ALT=<ID=INS:ME:ALU,Description="Insertion of ALU element">\n')
    header.append('##ALT=<ID=INS:ME:L1,Description="Insertion of L1 element">\n')
    header.append('##ALT=<ID=INS:ME:SVA,Description="Insertion of SVA element">\n')
    header.append('##ALT=<ID=INS:ME:MER,Description="Insertion of MER element">\n')
    header.append('##ALT=<ID=INS:ME:LTR,Description="Insertion of LTR element">\n')
    header.append('##ALT=<ID=INS:ME:PolyA,Description="Insertion of PolyA element">\n')
    header.append('##ALT=<ID=INS:ME:HERV,Description="Insertion of HERV element">\n')
    header.append('##FILTER=<ID=VQLOW,Description="Quality not VQHIGH">\n')
    header.append('##FILTER=<ID=SQLOW,Description="Somatic quality not SQHIGH">\n')
    header.append('##FILTER=<ID=URR,Description="Too close to an underrepresented repeat">\n')
    header.append('##FILTER=<ID=MPCBT,Description="Mate pair count below 10">\n')
    header.append('##FILTER=<ID=SHORT,Description="Junction side length below 70">\n')
    header.append('##FILTER=<ID=TSNR,Description="Transition sequence not resolved">\n')
    header.append('##FILTER=<ID=INTERBL,Description="Interchromosomal junction in baseline">\n')
    header.append('##FILTER=<ID=sns75,Description="Sensitivity to known MEI calls in range (.75,.95] i.e. medium FDR">\n')
    header.append('##FILTER=<ID=sns95,Description="Sensitivity to known MEI calls in range (.95,1.00] i.e. high to very high FDR">\n')
    header.append('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant described in this record">\n')
    header.append('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
    header.append('##INFO=<ID=IMPRECISE,Number=0,Type=Flag,Description="Imprecise structural variation">\n')
    header.append('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">\n')
    header.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    header.append('##INFO=<ID=SF,Number=.,Type=String,Description="Source File (index to sourceFiles, f when filtered)">\n')
    header.append('##INFO=<ID=AC,Number=.,Type=Integer,Description="Allele count in genotypes">\n')
    header.append('##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">\n')
    header.append('##SnpEffVersion="5.2 (build 2023-09-29 06:17), by Pablo Cingolani"\n')
    header.append('##SnpEffCmd="SnpEff  -lof GRCh37.75 decoy.vcf "\n')
    header.append("##INFO=<ID=ANN,Number=.,Type=String,Description=\"Functional annotations: 'Allele | Annotation | Annotation_Impact | Gene_Name | Gene_ID | Feature_Type | Feature_ID | Transcript_BioType | Rank | HGVS.c | HGVS.p | cDNA.pos / cDNA.length | CDS.pos / CDS.length | AA.pos / AA.length | Distance | ERRORS / WARNINGS / INFO' \">\n")
    header.append("##INFO=<ID=LOF,Number=.,Type=String,Description=\"Predicted loss of function effects for this variant. Format: 'Gene_Name | Gene_ID | Number_of_transcripts_in_gene | Percent_of_transcripts_affected'\">\n")
    header.append("##INFO=<ID=NMD,Number=.,Type=String,Description=\"Predicted nonsense mediated decay effects for this variant. Format: 'Gene_Name | Gene_ID | Number_of_transcripts_in_gene | Percent_of_transcripts_affected'\">\n")

    col_header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
    header.append("\t".join(col_header) + "\n")

    # --- variants ---
    positions = sorted(rng.sample(range(start_pos, end_pos + 1), k=min(n_variants, end_pos - start_pos + 1)))
    body = []

    for pos in positions:
        ref, alts = rand_ref_alts()
        num_alts = len(alts)

        gts = [sample_gt(num_alts) for _ in samples]
        gts_for_calc = [gt if gt != "." else "./." for gt in gts]

        ac, an = compute_ac_an(gts_for_calc, num_alts)
        ann = make_ann(pos, ref, alts)

        info = f"AC={','.join(map(str, ac))};AN={an};ANN={ann}"

        row = [
            chrom,
            str(pos),
            ".",
            ref,
            ",".join(alts),
            ".",
            ".",
            info,
            "GT",
            *gts,
        ]
        body.append("\t".join(row) + "\n")

    with open("xe2.eff.vcf", "w") as f:
        f.write("".join(header) + "".join(body))

def generate_decoy_evolution():
    n_pairs = 5000
    readlen = 150

    def write(path, mate):
        with gzip.open(path, "wt") as f:
            for i in range(1, n_pairs+1):
                rid = f"@CONTROL_LIBRARY|FAILED|DO_NOT_USE|{i:08d}/{mate}"
                seq = "N" * readlen
                qual = "!" * readlen   # Phred 0
                f.write(rid + "\n")
                f.write(seq + "\n+\n")
                f.write(qual + "\n")

    write("control_library.fastq.gz", 1)
    print("Wrote control_library")


def generate_transcriptome_decoy(
    out_path = "transcriptome_short_error.fa",
    n: int = 2000,
    length: int = 25,
    zero_pad: int = 4,
):

    if n <= 0:
        raise ValueError(f"n must be > 0 (got {n})")
    if length <= 0:
        raise ValueError(f"length must be > 0 (got {length})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random()
    bases = "ACGT"

    lines = []
    for i in range(n):
        seq = "".join(rng.choice(bases) for _ in range(length))
        header = f"{i:0{zero_pad}d}"
        lines.append(f">{header}")
        lines.append(seq)

    out_path.write_text("\n".join(lines) + "\n")
    return out_path

def generate_cell_label_decoys(
) -> None:
    import pandas as pd
    df = pd.read_excel("~/bioagent-data/single-cell/reference/Cell_marker_Seq.xlsx")
    required = ["species", "tissue_class", "tissue_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    rng = random.Random()

    # "Plausible" species names that are real organisms but clearly wrong for a Human/Mouse adipose/muscle labeling task
    species_pool = [
        "Danio rerio (Zebrafish)",
        "Xenopus laevis (African clawed frog)",
        "Drosophila melanogaster (Fruit fly)",
        "Caenorhabditis elegans (Nematode)",
        "Strongylocentrotus purpuratus (Sea urchin)",
        "Ciona intestinalis (Sea squirt)",
        "Nematostella vectensis (Sea anemone)",
        "Octopus vulgaris (Common octopus)",
        "Aplysia californica (Sea hare)",
        "Gallus gallus (Chicken)",
        "Anolis carolinensis (Green anole)",
        "Ambystoma mexicanum (Axolotl)",
        "Branchiostoma lanceolatum (Amphioxus)",
        "Saccharomyces cerevisiae (Yeast)",
        "Arabidopsis thaliana (Thale cress)", 
        "Tardigrada sp. (Tardigrade)",
        "Homo neanderthalensis (Neanderthal)", 
        "Mammuthus primigenius (Woolly mammoth)", 
        "Thylacinus cynocephalus (Thylacine)",
    ]

    tissue_pairs = [
        ("Respiratory system", "Gill filament"),
        ("Respiratory system", "Gill lamella"),
        ("Digestive system", "Crop"),
        ("Digestive system", "Gizzard"),
        ("Digestive system", "Hepatopancreas"),
        ("Nervous system", "Antenna lobe"),
        ("Nervous system", "Mushroom body"),
        ("Visual system", "Compound eye"),
        ("Integumentary system", "Exoskeleton cuticle"),
        ("Immune system", "Hemolymph"),
        ("Circulatory system", "Dorsal vessel"),
        ("Reproductive system", "Cloaca"),
        ("Reproductive system", "Oviduct gland"),
        ("Musculoskeletal system", "Cuttlebone"),
        ("Musculoskeletal system", "Swim bladder wall"),
        ("Buoyancy organ", "Swim bladder"),
        ("Sensory organ", "Lateral line neuromast"),
        ("Specialized secretory organ", "Ink sac"),
        ("Specialized secretory organ", "Photophore"),
        ("Plant tissue", "Leaf mesophyll"),
        ("Plant tissue", "Root meristem"),
        ("Plant tissue", "Xylem vessel"),
        ("Plant tissue", "Phloem sieve tube"),
    ]

    def choose_different(pool, original):
        if len(pool) == 1:
            return pool[0]
        for _ in range(20):
            v = rng.choice(pool)
            if str(v) != str(original):
                return v
        return rng.choice(pool)

    new_species = []
    new_tclass = []
    new_ttype = []

    for _, row in df.iterrows():
        sp = choose_different(species_pool, row["species"])
        tclass, ttype = rng.choice(tissue_pairs)


        if tclass == "Plant tissue":
            sp = rng.choice([
                "Arabidopsis thaliana (Thale cress)",
                "Oryza sativa (Rice)",
                "Zea mays (Maize)",
            ])
        elif sp.startswith("Arabidopsis") or sp.startswith("Oryza") or sp.startswith("Zea"):
            tclass, ttype = rng.choice([p for p in tissue_pairs if p[0] == "Plant tissue"])

        new_species.append(sp)
        new_tclass.append(tclass)
        new_ttype.append(ttype)

    df["species"] = new_species
    df["tissue_class"] = new_tclass
    df["tissue_type"] = new_ttype

    with pd.ExcelWriter("cell_markers_2.xlsx", engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="data")

def generate_dolphin_decoy():
    # wget -O Delphinapterus_leucas.delLeu1.dna.toplevel.fa.gz \
    # https://ftp.ensembl.org/pub/release-115/fasta/delphinapterus_leucas/dna/Delphinapterus_leucas.ASM228892v3.dna.toplevel.fa.gz


if __name__ == "__main__":
    generate_decoy_mouse()
    generate_decoy_fibrosis()
    generate_decoy_evolution()
    generate_transcriptome_decoy()
    generate_cell_label_decoys()
    generate_dolphin_decoy()
