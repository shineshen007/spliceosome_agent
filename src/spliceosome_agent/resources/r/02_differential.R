# Frozen two-group model. Python validates IDs/units and does not author statistics.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop("Usage: differential.R JOB.json")
if(!requireNamespace("jsonlite",quietly=TRUE)) stop("Missing package: jsonlite")
job <- jsonlite::read_json(args[1],simplifyVector=TRUE)
if(!is.null(job$r_library)) .libPaths(c(job$r_library,.libPaths()))
set.seed(job$seed)
options(warn=1)
dir.create(job$output,recursive=TRUE,showWarnings=FALSE)
writeLines(capture.output(sessionInfo()),file.path(job$output,"sessionInfo_start.txt"))
x <- as.matrix(read.csv(job$matrix,row.names=1,check.names=FALSE))
storage.mode(x) <- "double"
meta <- read.csv(job$metadata,check.names=FALSE,stringsAsFactors=FALSE)
rownames(meta) <- meta$sample_id
stopifnot(identical(colnames(x),meta$sample_id))
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])
source(file.path(dirname(script),"03_design.R"))
prepared <- prepare_design(meta,job)
meta <- prepared$metadata; formula <- prepared$formula; design <- prepared$design
absorbed <- prepared$absorbed; coefficient <- ncol(design)
ids <- rownames(x)
status <- rep("tested",nrow(x))
finite <- rowSums(!is.finite(x))==0L
status[!finite] <- "not_tested_missing"
if(job$missing_policy=="error" && any(!finite)) stop("Missing measurements; policy=error")
keep <- finite
if(job$scale=="counts") {
 if(any(x[finite,,drop=FALSE]<0) || any(x[finite,,drop=FALSE]!=floor(x[finite,,drop=FALSE]))) stop("Counts must be nonnegative integers")
 low <- rowSums(x,na.rm=TRUE)<job$counts_min_total
 status[finite & low] <- "not_tested_low_total_count"
 keep <- keep & !low
}
constant <- apply(x,1,function(v) all(is.finite(v)) && length(unique(v))<2L)
status[keep & constant] <- "not_tested_constant"
keep <- keep & !constant
res <- data.frame(feature_id=ids,effect=NA_real_,se=NA_real_,ci_low=NA_real_,ci_high=NA_real_,
                  statistic=NA_real_,pvalue=NA_real_,padj=NA_real_,status=status,n_units=if(job$design=="paired") nlevels(meta$subject) else nrow(meta))
normalized <- x
if(job$scale=="counts") normalized[,] <- NA_real_
method <- if(job$scale=="counts") "DESeq2_Wald" else "limma_moderated_t"
if(any(keep)) {
 if(job$scale=="counts") {
  if(!requireNamespace("DESeq2",quietly=TRUE)) stop("Missing package: DESeq2")
  dds <- DESeq2::DESeqDataSetFromMatrix(x[keep,,drop=FALSE],meta,formula)
  dds <- DESeq2::DESeq(dds,fitType="parametric",minReplicatesForReplace=Inf,parallel=FALSE)
  fitted <- DESeq2::results(dds,contrast=c("condition",job$case,job$reference),
                            alpha=job$alpha,independentFiltering=FALSE)
  r <- as.data.frame(fitted)
  bad <- !is.na(S4Vectors::mcols(dds)$betaConv) & !S4Vectors::mcols(dds)$betaConv
  r[bad,c("log2FoldChange","lfcSE","stat","pvalue")] <- NA_real_
  crit <- qnorm(1-(1-job$confidence)/2)
  res[keep,c("effect","se","ci_low","ci_high","statistic","pvalue")] <-
      cbind(r$log2FoldChange,r$lfcSE,r$log2FoldChange-crit*r$lfcSE,r$log2FoldChange+crit*r$lfcSE,r$stat,r$pvalue)
  res$status[keep] <- ifelse(bad,"not_tested_nonconverged",ifelse(is.finite(r$pvalue),"tested","not_tested_outlier"))
  normalized[keep,] <- log2(DESeq2::counts(dds,normalized=TRUE)+1)
  diag <- S4Vectors::mcols(dds)
  fields <- intersect(c("betaConv","betaIter","dispGeneIter","dispIter"),colnames(diag))
  write.csv(data.frame(feature_id=rownames(dds),as.data.frame(diag[,fields,drop=FALSE])),file.path(job$output,"model_diagnostics.csv"),row.names=FALSE)
 } else {
  if(!requireNamespace("limma",quietly=TRUE)) stop("Missing package: limma")
  fit <- limma::eBayes(limma::lmFit(x[keep,,drop=FALSE],design),trend=FALSE,robust=FALSE)
  se <- fit$stdev.unscaled[,coefficient]*sqrt(fit$s2.post)
  critical <- qt(1-(1-job$confidence)/2,fit$df.total)
  effect <- fit$coefficients[,coefficient]
  res[keep,c("effect","se","ci_low","ci_high","statistic","pvalue")] <-
     cbind(effect,se,effect-critical*se,effect+critical*se,fit$t[,coefficient],fit$p.value[,coefficient])
 }
}
tested <- is.finite(res$pvalue)
res$padj[tested] <- p.adjust(res$pvalue[tested],method="BH")
res$status[res$status=="tested" & !tested] <- "not_estimable"
res$selected <- is.finite(res$padj) & res$padj<=job$alpha & abs(res$effect)>=job$min_effect
res$direction <- ifelse(is.na(res$effect),"not_estimated",ifelse(res$effect>0,"up",ifelse(res$effect<0,"down","zero")))
res$effect_scale <- if(job$scale=="counts") "log2_fold_change" else "difference_on_declared_log2_scale"
res$method <- method; res$contrast <- paste0(job$case,"_minus_",job$reference)
write.csv(res,file.path(job$output,"differential.csv"),row.names=FALSE,na="")
write.csv(data.frame(feature_id=ids,normalized,check.names=FALSE),file.path(job$output,"display_expression.csv"),row.names=FALSE,na="")
write.csv(data.frame(sample_id=meta$sample_id,group=meta$group,subject_id=meta$subject_id,
 median=apply(normalized,2,median,na.rm=TRUE)),file.path(job$output,"sample_qc.csv"),row.names=FALSE)
write.csv(design,file.path(job$output,"design_matrix.csv"))
jsonlite::write_json(list(method=method,formula=paste(deparse(formula),collapse=""),
 absorbed_subject_covariates=as.list(absorbed),
 n_features=nrow(res),n_tested=sum(tested),n_selected=sum(res$selected),
 bh_scope="all_finite_pvalues_before_candidate_filter",independent_filtering=FALSE,
 design=job$design,n_samples=nrow(meta),n_units=if(job$design=="paired")nlevels(meta$subject)else nrow(meta)),
 file.path(job$output,"statistics.json"),auto_unbox=TRUE,pretty=TRUE)
writeLines(capture.output(sessionInfo()),file.path(job$output,"sessionInfo.txt"))
