# Load the required packages
library(haven)
library(tidyr)
library(dplyr)
library(magrittr)

# Set the working directory accordingly
# setwd("C:/Users/.../karnataka_survey") # nolint

DEFAULT_METADATA_FILEPATH <- "datav4.0/Data/2. Demographics and Outcomes/individual_characteristics.dta" # nolint
VALID_VILLAGE_IDS <- c(1:12, 14:21, 23:77) # village IDs 13 and 22 are missing
RAW_CSV_FOLDER <- "2010-0760_Data/Data/Raw_csv"

ties_layer_mapping <- list(borrowmoney = "money",
                          lendmoney = "money",
                          giveadvice = "advice",
                          helpdecision = "advice",
                          keroricego = "kerorice",
                          keroricecome = "kerorice",
                          visitgo = "visit",
                          visitcome = "visit")

get_karnataka_survey_data <- function(
                                village_id,
                                tie_type,
                                indivinfo,
                                ties_layer_mapping = list(
                                    borrowmoney = "money",
                                    lendmoney = "money",
                                    giveadvice = "advice",
                                    helpdecision = "advice",
                                    keroricego = "kerorice",
                                    keroricecome = "kerorice",
                                    visitgo = "visit",
                                    visitcome = "visit"
                                  ),
                                all_na_codes=c("9999999", "5555555", "7777777", "0"), # nolint
                                raw_csv_folder=RAW_CSV_FOLDER) {

  # Filter the individual-level metadata to keep only the relevant village
  resp <- subset(indivinfo, indivinfo$village == village_id)
  resp$didsurv <- 1

  village_file <- file.path(raw_csv_folder,
                            paste("village", village_id, ".csv", sep = ""))
  metadata <- read.csv(village_file, header = FALSE, as.is = TRUE)
  colnames(metadata) <- c("hhid", "ppid", "gender", "age")

  ## gender (1-Male, 2-Female)
  metadata$gender <- dplyr::recode(metadata$gender, "Male", "Female")

  ## pre-process pid to match the format in the individual-level metadata
  metadata$pid <- ifelse(nchar(metadata$ppid) == 2,
                         paste(metadata$hhid, metadata$ppid, sep = ""),
                         paste(metadata$hhid, 0, metadata$ppid, sep = ""))

  ## Select only the relevant columns
  selected_cols <- c("pid", "resp_status", "religion", "caste", "didsurv") 
  metadata <- merge(metadata,
                    resp[, selected_cols],
                    by = "pid",
                    all.x = TRUE,
                    all.y = TRUE)

  filepath <- file.path(raw_csv_folder,
                        paste(tie_type, village_id, ".csv", sep = ""))
  df_raw <- read.csv(filepath, header = FALSE, as.is = TRUE, na = all_na_codes)

  edgelist <- tidyr::pivot_longer(df_raw, cols = !V1, values_drop_na = TRUE)

  edgelist <- edgelist %>%
      dplyr::select(-name) %>%
      dplyr::rename(ego = V1, alter = value) %>%
      dplyr::mutate(reporter = ego)

  # Let's also add a column for the tie type
  edgelist$tie_type <- tie_type

  # Let's add a weight column too
  edgelist$weight <- 1

  # If the question was "Did you borrow money from anyone?", 
  # then we need to flip the ego and alter columns
  if(tie_type %in% c("borrowmoney", "helpdecision", "keroricego", "visitgo")){
    edgelist <- edgelist %>% dplyr::rename(ego = alter, alter = ego)
  }

  # Create a layer column and reorder the columns 
  # to make it easier to work with VIMuRe later
  edgelist <- edgelist %>%
    mutate(layer = unlist(ties_layer_mapping[tie_type])) %>% 
    select(ego, alter, reporter, tie_type, layer, weight)

  #### Further pre-processing steps ####

  # Who could actually report on the ties?
  reporters <- metadata %>%
    dplyr::filter(didsurv == 1) %>%
    dplyr::pull(pid) %>%
    as.vector()
  nodes <- reporters %>% union(edgelist$ego) %>% union(edgelist$alter)

  # Only keep reports made by those who were MARKED as reporters in metadata CSV
  edgelist <- edgelist %>% dplyr::filter(reporter %in% reporters)

  # Remove self-loops
  edgelist <- edgelist %>% dplyr::filter(ego != alter)

  # Remove duplicates
  edgelist <- edgelist %>% dplyr::distinct()

  return(list(edgelist = edgelist, reporters = reporters))

}

get_layer <- function(village_id,
                      layer_name,
                      indivinfo,
                      raw_csv_folder = RAW_CSV_FOLDER) {

  tie_types <- list(
    money = c("borrowmoney", "lendmoney"),
    advice = c("giveadvice", "helpdecision"),
    kerorice = c("keroricego", "keroricecome"),
    visit = c("visitgo", "visitcome")
  )

  selected_tie_types <- tie_types[[layer_name]]


  edgelist <- data.frame()
  reporters <- c()

  for (tie_type in selected_tie_types){
    data <- get_karnataka_survey_data(village_id, tie_type, indivinfo,
                                      raw_csv_folder = raw_csv_folder)
    edgelist <- rbind(edgelist, data$edgelist)
    reporters <- union(reporters, data$reporters)
  }

  return(list(edgelist = edgelist, reporters = reporters))
}

get_indivinfo <- function(metadata_file = DEFAULT_METADATA_FILEPATH){
    indivinfo <- haven::read_dta(metadata_file)
    ## one individual (6109803) is repeated twice. Remove the duplicate
    indivinfo <- indivinfo[!duplicated(indivinfo$pid) == TRUE, ]
    return(indivinfo)
}
