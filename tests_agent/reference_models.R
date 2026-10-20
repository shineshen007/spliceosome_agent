# Independent standard-package reference, without sourcing the agent tool.
args <- commandArgs(trailingOnly=TRUE)
library(jsonlite)
job <- read_json(args[1],simplifyVector=TRUE)
x <- as.matrix(read.csv(job$matrix,row.names=1,check.names=FALSE))
md <- read.csv(job$metadata,stringsAsFactors=FALSE)
md$group <- factor(md$group,levels=c(job$reference,job$case))
md$subject <- factor(md$subject_id)
rownames(md) <- md$sample_id
if(job$scale=="counts") {
  library(DESeq2)
  keep <- rowSums(x)>=job$counts_min_total & apply(x,1,sd)>0
  f <- if(job$design=="paired") ~subject+group else ~group
  dds <- DESeqDataSetFromMatrix(x[keep,,drop=FALSE],md,f)
  dds <- DESeq(dds,fitType="parametric",minReplicatesForReplace=Inf,parallel=FALSE)
  r <- as.data.frame(results(dds,contrast=c("group",job$case,job$reference),independentFiltering=FALSE,alpha=job$alpha))
  critical <- qnorm(1-(1-job$confidence)/2)
  out <- data.frame(feature_id=rownames(r),effect=r$log2FoldChange,se=r$lfcSE,pvalue=r$pvalue,
    ci_low=r$log2FoldChange-critical*r$lfcSE,ci_high=r$log2FoldChange+critical*r$lfcSE)
} else {
  library(limma)
  if(job$design=="paired") {
    pairs <- split(md,md$subject_id)
    delta <- sapply(pairs,function(p) x[,p$sample_id[p$group==job$case]]-x[,p$sample_id[p$group==job$reference]])
    fit <- eBayes(lmFit(delta,matrix(1,ncol(delta),1)),trend=FALSE,robust=FALSE)
    j <- 1L
  } else {
    design <- model.matrix(~group,md)
    fit <- eBayes(lmFit(x,design),trend=FALSE,robust=FALSE)
    j <- 2L
  }
  se <- fit$stdev.unscaled[,j]*sqrt(fit$s2.post)
  critical <- qt(1-(1-job$confidence)/2,fit$df.total)
  out <- data.frame(feature_id=rownames(x),effect=fit$coefficients[,j],se=se,pvalue=fit$p.value[,j],
    ci_low=fit$coefficients[,j]-critical*se,ci_high=fit$coefficients[,j]+critical*se)
}
out$padj <- p.adjust(out$pvalue,"BH")
write.csv(out,args[2],row.names=FALSE)

