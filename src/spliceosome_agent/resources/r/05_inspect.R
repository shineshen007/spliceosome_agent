# Bootstrap inspection uses base R only; it does not need a complete study or jsonlite.
args <- commandArgs(TRUE)
x <- readRDS(args[1])
if (!is.matrix(x) && !is.data.frame(x)) stop("Only matrix/data.frame RDS inputs are supported")
if (is.null(rownames(x))) stop("Explicit row names are required")
if (anyDuplicated(rownames(x)) || anyDuplicated(colnames(x))) stop("Duplicate input identifiers")
utils::write.csv(data.frame(feature_id=rownames(x),x,check.names=FALSE),args[2],row.names=FALSE,na="")
keys <- c("scale","preprocessing","species","id_type")
values <- vapply(keys,function(k) {
 v <- attr(x,k,exact=TRUE)
 if(is.character(v) && length(v)==1L && !is.na(v) && nchar(v)<=512L) v else ""
},character(1))
utils::write.csv(data.frame(key=keys,value=values),args[3],row.names=FALSE)
