# Explicit, project-local statistical environment. Never install into a system library.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop("Usage: Rscript install.R TARGET_LIBRARY")
lib <- path.expand(args[1])
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
lib <- normalizePath(lib, mustWork=TRUE)
.libPaths(c(lib, .libPaths()))
if (getRversion() < "4.5" || getRversion() >= "4.6") stop("Locked environment requires R 4.5.x / Bioconductor 3.22")
options(repos=c(CRAN="https://cloud.r-project.org"), timeout=600)
kind <- if (Sys.info()[["sysname"]] == "Darwin") "binary" else "source"
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", lib=lib, type=kind)
BiocManager::install(c("jsonlite", "limma", "DESeq2"), lib=lib, version="3.22", ask=FALSE, update=FALSE, type=kind)
script <- sub("^--file=", "", grep("^--file=",commandArgs(),value=TRUE)[1])
lock <- file.path(dirname(normalizePath(script)),"versions.lock.json")
if(file.exists(lock)) {
 expected <- jsonlite::read_json(lock,simplifyVector=TRUE)$packages
 drift <- names(expected)[vapply(names(expected),function(p) !requireNamespace(p,quietly=TRUE) || as.character(packageVersion(p))!=expected[[p]],logical(1))]
 if(length(drift)) stop("Locked R environment mismatch: ",paste(drift,collapse=", "),". Preserve this library and restore the recorded versions; do not silently update an existing run.")
}
packages <- installed.packages(lib.loc=lib)
write.csv(packages[,c("Package","Version","Built"),drop=FALSE], file.path(lib,"installed_versions.csv"), row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(lib,"sessionInfo.txt"))
