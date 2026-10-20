# Shared design construction: no fitting or significance tests.
prepare_design <- function(meta, job) {
meta$condition <- factor(meta$group,levels=c(job$reference,job$case))
meta$subject <- factor(meta$subject_id)
covnames <- character()
absorbed <- character()
if(length(job$covariates)) for(i in seq_along(job$covariates)) {
  old <- names(job$covariates)[i]; new <- paste0("cov",i)
  meta[[new]] <- if(job$covariates[[old]]=="continuous") as.numeric(meta[[old]]) else factor(meta[[old]])
  # Subject-fixed effects absorb between-subject covariates in paired models.
  if(job$design=="paired" && all(vapply(split(meta[[new]],meta$subject),function(v)length(unique(v))==1L,logical(1)))) {
    absorbed <- c(absorbed,old)
    next
  }
  covnames <- c(covnames,new)
}
terms <- c(if(job$design=="paired") "subject",covnames,"condition")
formula <- reformulate(terms)
design <- model.matrix(formula,meta)
if(qr(design)$rank!=ncol(design)) stop("Design is rank deficient: confounded or constant covariates")
if(nrow(design)<=ncol(design)) stop("No residual degrees of freedom")
list(metadata=meta,formula=formula,design=design,absorbed=absorbed)
}
