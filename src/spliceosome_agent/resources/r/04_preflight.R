# Structural validation only; never fits a statistical model.
args <- commandArgs(TRUE)
job <- jsonlite::read_json(args[1],simplifyVector=TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])
source(file.path(dirname(script),"03_design.R"))
meta <- read.csv(job$metadata,check.names=FALSE,stringsAsFactors=FALSE)
prepared <- prepare_design(meta,job)
jsonlite::write_json(list(rank=qr(prepared$design)$rank,columns=ncol(prepared$design),
 residual_df=nrow(prepared$design)-ncol(prepared$design),
 absorbed_subject_covariates=as.list(prepared$absorbed)),job$output,auto_unbox=TRUE,pretty=TRUE)
