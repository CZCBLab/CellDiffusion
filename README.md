![CellDiffusion](https://github.com/CZCBLab/CellDiffusion/blob/main/CellDiffusion_1.pdf)

# About:
Single-cell RNA sequencing data integration is crucial but remains challenging, particularly in the presence of cell-type-specific batch effects and critical biological variations across batches, 
such as imbalanced cell compositions and non-overlapping cell types. Despite hundreds of analysis tools developed over the past decade, robust solutions to overcome this challenge are still lacking. 
We present CellDiffusion, a physics-inspired, graph-based diffusion model designed to optimize cell-cell associations across batches through feature diffusion. By learning a cell-cell attention network, 
CellDiffusion adaptively strengthens biologically meaningful connections. Our approach applies cell-specific correction in data integration, thereby aligning cells across batches while preserving intrinsic 
cell types and developmental trajectories. We validate CellDiffusion across various datasets, including the human endometrium, one of the most dynamic and complex tissues featuring extensive cellular and 
molecular changes with the menstrual cycle. This framework also improves other essential analyses such as label transfer, clustering, and trajectory inference, representing a powerful paradigm for accurate 
single-cell transcriptomic integrative analysis.


This repository contains the source code for the paper: "CellDiffusion: Enhancing single-cell data integration with accurate inference of cell-cell association", 
Yu-Chen Liu#, Ziying Huang#, Simon Liang Lu, Lei Jiang, Zedong Lin, Jou-Hsuan Lee, Vijaya B. Kolachalama, Rong Li*, Dong Xu*, Juexin Wang*, Heng Pan*, & Chao Zhang*, in preparation.

![CellDiffusion](https://github.com/CZCBLab/CellDiffusion/blob/main/CellDiffusion_2.pdf)
