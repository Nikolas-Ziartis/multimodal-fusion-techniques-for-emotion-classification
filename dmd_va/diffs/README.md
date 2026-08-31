# Minimality proof: official DMD -> DMD-VA

Each *.diff is `diff -u <official> <DMD-VA>` (line endings normalised so only
content edits show). Originals are the files from https://github.com/mdswyz/DMD :
    dmd_model.diff    <- trains/singleTask/model/dmd.py
    dmd_trainer.diff  <- trains/singleTask/DMD.py
    hetero_kernel.diff<- trains/singleTask/distillnets/get_distillation_kernel.py
    homo_kernel.diff  <- trains/singleTask/distillnets/get_distillation_kernel_homo.py
    misc.diff         <- trains/singleTask/utils/misc.py
    hinge_loss.diff   <- trains/singleTask/.../HingeLoss.py
Every edited line is tagged `# DMD-VA:` in the source for in-place visibility.
