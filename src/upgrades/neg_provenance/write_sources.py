#!/usr/bin/env python3
"""Primary-source provenance table for the negative sets of human_enhancers_cohn and nt_enhancers.

Every quote below was re-read on 2026-09-26 from the source named in `verified_via` (not copied from the
interrupted first run; fix round 1 added the isochore-boundary rows and the independent-set header row). Quotes are verbatim; [..] marks an omission, and the S-/S+ subscripts of Liu et al.
are rendered in plain text. Writes results/upgrades/neg_provenance_sources.csv.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402

D = "2026-09-26"
COHN_REPO = "https://github.com/cohnDikla/enhancer_CNN (commit 7dcab48de4a6b1a09c81467a1f0cb2b47b073d0c)"
GB_REPO = "https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks (commit 605d8539830e16c85abe7826990958303ffc5e1c)"
NT_CARD = ("https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks "
           "(README.md at pinned revision 96d86d567d4cd33536e49b429dc7983121619a08)")

ROWS = [
    # ------------------------------------------------------------------ human_enhancers_cohn
    ("cohn", "negative sampling rule", "Cohn, Zuk & Kaplan 2018, bioRxiv, sec. 2.3.1",
     "doi:10.1101/264200 ; https://www.biorxiv.org/content/10.1101/264200v1.full",
     "For each species, we sampled a random set of genomic regions (500 bp long), excluding regions with N’s, "
     "regions less than 15Kb from genes (Zerbino et al., 2018), and H3K27ac bound regions (Villar et al., 2015).",
     "WebFetch of bioRxiv full text", "Stated in the single-TF section; sec. 2.4.1 re-uses 'biological non-enhancer "
     "sequences' for the H3K27ac enhancer task. Zerbino et al. 2018 = Ensembl 2018 (gene set; biotypes not stated). "
     "No GC/length matching other than the fixed 500 bp is described."),
    ("cohn", "negative sampling rule (code)", "Cohn et al. code, H3K27ac_vs_negative_data/DataLoaderH3K27acvsNeg.py",
     COHN_REPO,
     "The negative samples are already randomly chosen from the genome of each species, such that they are not "
     "intersected with any gene (+-15000 bases in each direction) and with any peak of H3K27ac.",
     "gh api + raw fetch at the commit; byte-identical to the copy in external/neg_provenance/sources",
     "The script that drew the negatives ('_new_negative_samples.txt') is not in the repository."),
    ("cohn", "positives", "Cohn, Zuk & Kaplan 2018, bioRxiv, sec. 3.3",
     "doi:10.1101/264200",
     "We used data from Villar et al., who tracked the evolution of promoters and active enhancers across the "
     "livers of 20 mammalian species (Villar et al., 2015). [..] we extracted 14K peaks as positive samples for each species.",
     "WebFetch of bioRxiv full text", "Positives = human-liver H3K27ac peaks (Villar et al. 2015, Cell 160:554)."),
    ("cohn", "set size", "Cohn, Zuk & Kaplan 2018, bioRxiv, sec. 2.4",
     "doi:10.1101/264200", "We extracted 14K positive and negative samples for each species.",
     "WebFetch of bioRxiv full text", ""),
    ("cohn", "released files", "Cohn et al. code README",
     COHN_REPO,
     "For each of the 17 species we used in the paper, you can find the positive_samples and negative_samples "
     "sequences (500bp each) in appropriate files, 14K in each file.",
     "raw fetch at the commit (identical to saved copy)",
     "Archive http://www.cs.huji.ac.il/~tommy//enhancer_CNN/Enhancers_vs_negative.tgz returned a server error page on "
     "2026-09-26, so the original (pre-GB) build/coordinates of the negatives could not be inspected."),
    ("cohn", "GB import + genome build", "Genomic Benchmarks docs/human_enhancers_cohn/README.md",
     GB_REPO,
     "The data have been imported with `seq2loc` utility. Totally, 13895 out of 14000 human enhancers and 13896 out of "
     "140000 negative controls have been found in the human genome reference and included.",
     "raw fetch at the commit (identical to saved copy)",
     "'140000' is a typo in the source (14000). create_datasets.ipynb: reference = Ensembl release-97 "
     "Homo_sapiens.GRCh38.dna.toplevel; fasta2loc exact matching; '27791 sequences found in the reference'; "
     "train_test_split(shuffle=True, random_state=42). The four interval files are byte-identical to localization/cohn/."),
    ("cohn", "GB import", "Grešová et al. 2023, BMC Genomic Data 24:25, 'Data format'",
     "doi:10.1186/s12863-023-01123-8 (PMC10150520)",
     "All samples were stored as genomic coordinates, and datasets originally provided as sequences (human enhancers "
     "Cohn, human nonTATA promoters) were mapped to the reference using the ‘seq2loc‘ tool included in the package.",
     "PubMed Central full text (PubMed MCP)", "GB did not re-sample Cohn's negatives; they are Cohn's own."),
    # ------------------------------------------------------------------ nt_enhancers
    ("nt_enhancers", "dataset composition", "Nucleotide Transformer downstream-tasks dataset card",
     NT_CARD,
     "To build the training dataset, the authors collect 742 strong enhancers, 742 weak enhancers and 1484 "
     "non-enhancers, and augment the dataset with 6000 synthetic enhancers and 6000 synthetic non-enhancers produced "
     "with a generative model. The test dataset is comprised of 100 strong enhancers, 100 weak enhancers and 200 non enhancers.",
     "curl of README.md at the pinned revision", "Source paper named on the card: Geng et al. 2022."),
    ("nt_enhancers", "synthetic augmentation", "Geng, Yang & Zhang 2022, Biophys Chem 286:106822, abstract",
     "doi:10.1016/j.bpc.2022.106822 (PMID 35605495)",
     "Firstly, considering the small number of sequences in the benchmark dataset, RankGAN is used to amplify the "
     "dataset size while maintaining the data characteristics.",
     "PubMed record", "Full text is paywalled; the GAN's training data and length handling are not verifiable here."),
    ("nt_enhancers", "genomic negatives: source", "Liu et al. 2016 (iEnhancer-2L), Bioinformatics 32:362, sec. 2.1",
     "doi:10.1093/bioinformatics/btv604 (PMID 26476782)",
     "The benchmark dataset was constructed based on the chromatin state information of nine cell lines, including "
     "H1ES, K562, GM12878, HepG2, HUVEC, HSMM, NHLF, NHEK and HMEC. The chromatin state information was annotated by "
     "ChromHMM (Ernst and Kellis, 2012; Ernst et al., 2011) with the whole genome profile of multiple histone marks, "
     "such as H3K4me1, H3K4me3, H3K27ac, etc.",
     "WebFetch of academic.oup.com article page", "Ernst et al. 2011 Nature 473:43 (doi:10.1038/nature09906)."),
    ("nt_enhancers", "genomic negatives: rule", "Liu et al. 2016 (iEnhancer-2L), sec. 2.1",
     "doi:10.1093/bioinformatics/btv604",
     "The strong enhancers in S strong+ and the non-enhancers in S− are clearly annotated in all the nine tissues "
     "mentioned above, but the weak enhancers in S weak+ vary significantly with different tissues.",
     "WebFetch of academic.oup.com article page",
     "Which ChromHMM states count as 'non-enhancer' is not enumerated; genome build is not stated in the text."),
    ("nt_enhancers", "genomic negatives: counts", "Liu et al. 2016 (iEnhancer-2L), sec. 2.1",
     "doi:10.1093/bioinformatics/btv604",
     "Thus, we obtained 742 strong enhancers, 370 517 weak enhancers, and 5 257 994 non-enhancers. [..] "
     "To get a benchmark dataset with balanced subsets in size, we randomly picked 742 samples from the weak "
     "enhancers and 1484 samples from the non-enhancers.",
     "WebFetch of academic.oup.com article page", "Negatives = a random 1484 of 5.26 M non-enhancer fragments."),
    ("nt_enhancers", "fragmentation / redundancy", "Liu et al. 2016 (iEnhancer-2L), sec. 2.1",
     "doi:10.1093/bioinformatics/btv604",
     "All the samples obtained via the above procedure were divided into 200 bp fragments in order to be consistent "
     "with the length of nucleosome and linker DNA. [..] the CD-HIT software (Li and Godzik, 2006) was used by setting "
     "the cutoff threshold at 80% to remove those DNA fragments with high sequence similarity",
     "WebFetch of academic.oup.com article page", ""),
    ("nt_enhancers", "independent test set", "Liu et al. 2018 (iEnhancer-EL), Bioinformatics 34:3835",
     "doi:10.1093/bioinformatics/bty458 (PMID 29878118)",
     "An independent dataset was used to further evaluate the performance of various methods, which was constructed "
     "based on the same protocol as the one used in constructing the benchmark dataset. The independent dataset "
     "contains 100 strong enhancers, 100 weak enhancers and 200 non-enhancers",
     "WebFetch of academic.oup.com article page", "NT test split = this independent set (400 sequences)."),
    ("nt_enhancers", "sequence files + build", "iEnhancer-2L benchmark FASTA (two community copies)",
     "https://github.com/lhy0322/iEnhancer-BERT (commit 6a54f6db5e6b1a71213efcea0f9bb06b796e9de7); "
     "https://github.com/chen-bioinfo/iEnhancer-ELM (commit f499f95984d5950f6c799204c4132f38f2776897)",
     "training headers of the form 'CHRX_10705600_10705800'; independent-set headers of the form "
     "'hg19_ct_UserTrack_3545_...'",
     "gh api + raw fetch at the commits", "The two copies are identical as sequence sets with the published counts "
     "(742/742/1484; 100/100/200). Build hg19 is confirmed empirically (neg_provenance_meta.csv nt_hdr_*)."),
    ("nt_enhancers", "independent-test negatives: parent regions", "iEnhancer-ELM copy, elm_original_non_enhancer_ind.txt",
     "https://github.com/chen-bioinfo/iEnhancer-ELM (commit f499f95984d5950f6c799204c4132f38f2776897)",
     "hg19_ct_UserTrack_3545_158rangechr1278319870-783938695pad0",
     "raw file at the commit (external/neg_provenance/ienhancer/)",
     "A UCSC Table-Browser FASTA header of a hg19 custom track with punctuation stripped, i.e. "
     "'hg19_ct_UserTrack_3545_158 range=chr12:78319870-78393869 5'pad=0 ...': each independent-test negative was "
     "cut from a larger user-track region. Analysed in nt_split_provenance.py (neg_provenance_nt_test_parent.csv). "
     "The training non-enhancers instead carry their own 200-bp hg19 window ('CHRX_10705600_10705800')."),
    # ------------------------------------------------------------------ annotation resources used
    ("both", "annotation", "GENCODE v44 comprehensive GTF (GRCh38)",
     "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz",
     "##description: evidence-based annotation of the human genome (GRCh38), version 44 (Ensembl 110)",
     "file header (verify_inputs.py)", ""),
    ("both", "annotation", "UCSC hg38 RepeatMasker, gap table, liftOver chains",
     "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/{rmsk,gap}.txt.gz ; "
     "https://hgdownload.soe.ucsc.edu/goldenPath/{hg38/liftOver/hg38ToHg19,hg19/liftOver/hg19ToHg38}.over.chain.gz",
     "", "verify_inputs.py (soft-mask concordance, N-content, chain headers)", ""),
    ("both", "annotation", "ENCODE SCREEN Registry V4 cCREs (GRCh38)",
     "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed", "", "verify_inputs.py (2,348,854 EH38E records)", ""),
    ("both", "annotation", "Broad/ENCODE ChromHMM 15-state segmentations, nine cell lines (hg19)",
     "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHmm/ ; doi:10.1038/nature09906", "",
     "verify_inputs.py (hg19 coordinates, 15 states, 9 cell lines)", "The same nine cell lines iEnhancer-2L used."),
    ("both", "ChromHMM marks and state 13", "UCSC track description, ENC Histone Broad ChromHMM (wgEncodeBroadHmm), hg19",
     "https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg19&g=wgEncodeBroadHmm",
     "CTCF, H3K4me1, H3K4me2, H3K4me3, H3K27ac, H3K9ac, H3K36me3, H4K20me1, H3K27me3, Input [..] Heterochromatin; "
     "low signal",
     "WebFetch of the UCSC track page", "State 13 is the low-signal state of a nine-mark model without H3K9me3, so "
     "'state 13' is described here as unmarked/low-signal chromatin, not as constitutive heterochromatin. The "
     "segmentation was made on hg18 and lifted to hg19 by UCSC."),
    ("both", "isochore classes", "Costantini et al. 2006, Genome Res 16:536",
     "doi:10.1101/gr.4910606 (PMID 16597586)",
     "Isochores visibly belong to five families characterized by different GC levels, as expected from previous "
     "investigations. [abstract]",
     "PubMed record",
     "The Costantini full text is not retrievable here (publisher login; PMC1457033 full text not served). The "
     "family boundaries and the 100-kb window are confirmed by the two open-access rows below, which cite it."),
    ("both", "isochore family boundaries", "Schmidt & Frishman 2008, Genome Biol 9:R104, Materials and methods",
     "doi:10.1186/gb-2008-9-6-r104 (PMC2481423)",
     "Bernardi and colleagues [18] defined the isochores according to their GC content. There are three isochore "
     "types with high GC content, H3 (>53%), H2 (46-53%), and H1 (41-46%), and two types with low GC content, L1 "
     "(<37%) and L2 (37-41%).",
     "Europe PMC full-text XML (saved: external/neg_provenance/sources/schmidt_frishman2008_PMC2481423.xml)",
     "Ref. [18] = Costantini et al. 2006. These are exactly the boundaries used here (L1<37, L2 37-41, H1 41-46, "
     "H2 46-53, H3>53 %GC)."),
    ("both", "isochore window size + boundaries", "Cozzi, Milanesi & Bernardi 2015, Evol Bioinform 11:253",
     "doi:10.4137/EBO.S27693 (PMC4662427)",
     "Costantini et al. 15 , using UCSC release hg17, partitioned the entire chromosomal sequences of the human "
     "genome 16 – 18 assembly into nonoverlapping 100-kb windows [..] local maxima (peaks) of isochore families at "
     "35.5%, 38.7%, 43.0%, 48.5%, and 55.0% GC for L1, L2, H1, H2, and H3 isochores and local minima (valleys) at "
     "37%, 41%, 46%, and 53%",
     "Europe PMC full-text XML (saved: external/neg_provenance/sources/cozzi2015_PMC4662427.xml)",
     "Confirms the 100-kb window and the 37/41/46/53% family boundaries. Our 100-kb window is centred on each "
     "query window (not a fixed tiling), a minor deviation."),
    ("both", "isochore biology", "Arhondakis, Auletta & Bernardi 2011, Genome Biol Evol 3:1080",
     "doi:10.1093/gbe/evr017 (PMC3227402)",
     "gene-poor regions, the “genome desert,” corresponding to the isochores of the L1, L2, and H1 families, which "
     "represent 85% of the genome [..] regulatory sequences predominantly belong in a “TATA-box” model in L1 and in a "
     "“GC-rich” model in H3",
     "PubMed Central full text (PubMed MCP)", "Background for the gene-desert -> AT-rich isochore mechanism."),
]


def main():
    df = pd.DataFrame(ROWS, columns=["dataset", "topic", "source", "doi_or_url", "quote", "verified_via", "note"])
    df.insert(6, "verified_on", D)
    out = os.path.join(C.RESULTS_DIR, "neg_provenance_sources.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
