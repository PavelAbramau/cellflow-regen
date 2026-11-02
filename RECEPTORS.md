# RECEPTORS — receptor layer (notebook 21)

Built 2026-11-01 by [notebooks/21_receptor_layer.ipynb](notebooks/21_receptor_layer.ipynb). CPU only, no model code.

## Sources and roles
* **Binding B:** OmniPath `LigRecExtra` (omnipath 1.0.12, `license="academic"`). Complexes come from the curated trio **CellPhoneDB, CellChatDB, ICELLNET**; resource of origin is kept per edge (`trio_sources`, `all_sources`). Fallback (evidence C): other OmniPath resources, direct edges, `n_sources >= 3`.
* **Sign:** CellChatDB-cofactors inhibition edges (OmniPath). Antagonist = −1 on its own receptor subunits and on the receptors of the ligands it neutralises. Antagonists among the 90: **IL-1Ra, IL-36Ra, Noggin, Decorin**. Assertion that IL-1Ra and IL-36Ra come out negative passes.
* **NicheNet:** not used in this notebook (kept free for notebook 22).
* **Receptor abundance R:** Parse PBS pseudobulks (raw counts, all ~40k genes), 18 members of `pseudobulks_20260203.zip` (6.2 GB archive) fetched with HTTP range requests (117 MB transferred, 174 MB on disk in `data/parse_pseudobulk_PBS/`). **Licence CC BY-NC 4.0** (non-commercial use only).
* Decoys excluded from B and competence: IL1R2, IL13RA2, TNFRSF11B, TNFRSF6B, TNFRSF10C/D, IL22RA2 (my curation choice, flagged in `data/binding_edges.parquet`).

## Competence
Per (ligand, donor, cell type): each trio complex is an alternative receptor; its subunits are all required, so `competence_min` = max over complexes of the **minimum** log1p(CPM) over its subunits. `competence_mean` is the same with the mean (sensitivity). Example: IL-2 in B naive: min 0.90 vs mean 2.86.

## Coverage
**85 of 90 cytokines have annotated receptors.** Evidence: A (≥ 2 of trio) 83, B (1 of trio) 1, C (other OmniPath only) 1, none 5.
No annotation: **ADSF, IL-17D, IL-32-beta, LT-alpha1-beta2, LT-alpha2-beta1** (their rows in B are all-zero; D_binding fixes their distance to every other ligand to 1.0 — flag these when using D_binding). Subunits per cytokine: median 2, max 16 (Noggin).
Caveats: IFN-alpha1 annotated via alias IFNA13 (same protein); GFRA4 missing from the Parse gene annotation, so **PSPN competence is NaN**; Noggin's 16 subunits are inherited from BMP/GDF targets; Plasmablast has 11 donors.

| cytokine | gene(s) | family | subunits | complexes | evidence | trio resources | sign |
|---|---|---|---|---|---|---|---|
| 4-1BBL | TNFSF9 | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| ADSF | RETN | Other | 0 | 0 | none |  | none |
| APRIL | TNFSF13 | TNF | 3 | 3 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| BAFF | TNFSF13B | TNF | 3 | 3 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| C3a | C3 | Complement | 3 | 3 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| C5a | C5 | Complement | 2 | 2 | A | CellChatDB;CellPhoneDB | agonist (+1) |
| CD27L | CD70 | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| CD30L | TNFSF8 | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| CD40L | CD40LG | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| CT-1 | CTF1 | IL-6/IL-12 | 2 | 1 | A | CellChatDB;ICELLNET | agonist (+1) |
| Decorin | DCN | Other | 7 | 7 | B | ICELLNET | antagonist (-1) |
| EGF | EGF | Growth factor | 3 | 3 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| EPO | EPO | Growth factor | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| FGF-beta | FGF2 | Growth factor | 8 | 8 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| FLT3L | FLT3LG | Growth factor | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| FasL | FASLG | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| G-CSF | CSF3 | Growth factor | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| GDNF | GDNF | Other | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| GITRL | TNFSF18 | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| GM-CSF | CSF2 | Common β chain | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| HGF | HGF | Growth factor | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IFN-alpha1 | IFNA13 | Interferon | 2 | 1 | A | CellChatDB;ICELLNET | agonist (+1) |
| IFN-beta | IFNB1 | Interferon | 3 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IFN-epsilon | IFNE | Interferon | 3 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IFN-gamma | IFNG | Interferon | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IFN-lambda1 | IFNL1 | Interferon | 2 | 1 | A | CellPhoneDB;ICELLNET | agonist (+1) |
| IFN-lambda2 | IFNL2 | Interferon | 2 | 1 | A | CellPhoneDB;ICELLNET | agonist (+1) |
| IFN-lambda3 | IFNL3 | Interferon | 2 | 1 | A | CellPhoneDB;ICELLNET | agonist (+1) |
| IFN-omega | IFNW1 | Interferon | 3 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IGF-1 | IGF1 | Growth factor | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-1-alpha | IL1A | IL-1 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-1-beta | IL1B | IL-1 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-10 | IL10 | IL-10 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-11 | IL11 | IL-6/IL-12 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-12 | IL12A+IL12B | IL-6/IL-12 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-13 | IL13 | Common γ chain/IL-13/TSLP | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-15 | IL15 | Common γ chain/IL-13/TSLP | 3 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-16 | IL16 | Interleukin - non-classified | 1 | 1 | A | CellChatDB;ICELLNET | agonist (+1) |
| IL-17A | IL17A | IL-17 | 2 | 1 | A | CellChatDB;ICELLNET | agonist (+1) |
| IL-17B | IL17B | IL-17 | 1 | 1 | A | CellChatDB;ICELLNET | agonist (+1) |
| IL-17C | IL17C | IL-17 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-17D | IL17D | IL-17 | 0 | 0 | none |  | none |
| IL-17E | IL25 | IL-17 | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-17F | IL17F | IL-17 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-18 | IL18 | IL-1 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-19 | IL19 | IL-10 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-1Ra | IL1RN | IL-1 | 2 | 3 | A | CellPhoneDB;ICELLNET | antagonist (-1) |
| IL-2 | IL2 | Common γ chain/IL-13/TSLP | 3 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-20 | IL20 | IL-10 | 3 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-21 | IL21 | Common γ chain/IL-13/TSLP | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-22 | IL22 | IL-10 | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-23 | IL23A+IL12B | IL-6/IL-12 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-24 | IL24 | IL-10 | 3 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-26 | IL26 | IL-10 (IL-20 subfamily) | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-27 | IL27+EBI3 | IL-6/IL-12 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-3 | IL3 | Common β chain | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-31 | IL31 | IL-6/IL-12 | 2 | 1 | A | CellChatDB;ICELLNET | agonist (+1) |
| IL-32-beta | IL32 | Interleukin - non-classified | 0 | 0 | none |  | none |
| IL-33 | IL33 | IL-1 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-34 | IL34 | Growth factor | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-35 | IL12A+EBI3 | IL-6/IL-12 | 2 | 1 | A | CellPhoneDB;ICELLNET | agonist (+1) |
| IL-36-alpha | IL36A | IL-1 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-36Ra | IL36RN | IL-1 | 2 | 3 | A | CellPhoneDB;ICELLNET | antagonist (-1) |
| IL-4 | IL4 | Common γ chain/IL-13/TSLP | 3 | 3 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-5 | IL5 | Common β chain | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-6 | IL6 | IL-6/IL-12 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-7 | IL7 | Common γ chain/IL-13/TSLP | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-8 | CXCL8 | Chemokine | 4 | 4 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| IL-9 | IL9 | Common γ chain/IL-13/TSLP | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| LIF | LIF | IL-6/IL-12 | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| LIGHT | TNFSF14 | TNF | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| LT-alpha1-beta2 | LTA+LTB | TNF | 0 | 0 | none |  | none |
| LT-alpha2-beta1 | LTA+LTB | TNF | 0 | 0 | none |  | none |
| Leptin | LEP | Other | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| M-CSF | CSF1 | Growth factor | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| Noggin | NOG | Other | 16 | 16 | C |  | antagonist (-1) |
| OSM | OSM | IL-6/IL-12 | 3 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| OX40L | TNFSF4 | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| PRL | PRL | Other | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| PSPN | PSPN | Other | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| RANKL | TNFSF11 | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| SCF | KITLG | Growth factor | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| TGF-beta1 | TGFB1 | Other | 4 | 3 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| TL1A | TNFSF15 | TNF | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| TNF-alpha | TNF | TNF | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| TPO | THPO | Other | 1 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| TRAIL | TNFSF10 | TNF | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| TSLP | TSLP | Common γ chain/IL-13/TSLP | 2 | 1 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| TWEAK | TNFSF12 | TNF | 2 | 2 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |
| VEGF | VEGFA | Growth factor | 4 | 5 | A | CellChatDB;CellPhoneDB;ICELLNET | agonist (+1) |

## Files
`data/binding_B.parquet` (90 × subunits, signed), `data/binding_B_long.parquet`, `data/binding_edges.parquet`, `data/competence_R.parquet` (ligand × donor × cell type: competence_min/mean, limiting_subunit), `data/receptor_abundance_R.parquet`, `data/D_binding_signed.{npy,csv}`, `data/D_binding_unsigned.{npy,csv}`, `data/receptor_coverage.csv`, figures `models/21_coverage.png`, `models/21_D_binding.png`.
