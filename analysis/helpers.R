aoa_data <- function(language, form, input_measure) {
  instrument_data <- get_instrument_data(language, form, administration_info = TRUE) |>
    filter(!is.na(!!sym(input_measure)))
  aoa <- merge(x=fit_aoa(instrument_data, measure=input_measure, age_min = 12),y=get_item_data(language, form),by="item_id") |>
    filter(!is.na(aoa)) |>
    # Only using the first phrase of the description
    mutate(item = str_extract(item_definition, "^[^/()]+")) |>
    # Only count the AoA as the last learned date of that word if there are multiple versions of the word
    filter(aoa == max(aoa, na.rm = TRUE,), .by=item)
  return(aoa)
}
