# filtered_customers <- plyr::count(bankSim, c("customer", "fraud"))
# dupl_data_customers <- filtered_customers[duplicated(filtered_customers[, "customer"]), ]
# 
# #getting those customers that have exhibited fraud
# bankSim_filter_customers <- bankSim %>%
#   filter(customer %in% dupl_data_customers[, "customer"])  %>%
#   select(customer) %>%
#   distinct(customer) %>%
#   arrange(customer)
# 
# bankSim <- bankSim %>%
#   filter(customer %in% bankSim_filter_customers[, "customer"])
# rm(filtered_customers)
# rm(dupl_data_customers)
# rm(bankSim_filter_customers)
###################
# filtered_merchant <- plyr::count(bankSim, c("merchant", "fraud"))
# dupl_data_merchant <- filtered_merchant[duplicated(filtered_merchant[, "merchant"]), ]
# 
# #getting those merchants that have exhibited fraud
# bankSim_filter_merchant <- bankSim %>%
#   filter(merchant %in% dupl_data_merchant[, "merchant"])  %>%
#   select(merchant) %>%
#   distinct(merchant) %>%
#   arrange(merchant)
# 
# bankSim <- bankSim %>%
#   filter(merchant %in% bankSim_filter_merchant[, "merchant"])
# 
# rm(filtered_merchant)
# rm(dupl_data_merchant)
# rm(bankSim_filter_merchant)
###################################################################

# Removing manually - just for validating that automated removal works
# bankSim_filter <- bankSim %>%
#   filter(category != "'es_transportation'") %>%
#   filter(category != "'es_food'") %>%
#   filter(category != "'es_contents'") %>%
#   filter(gender != "'U'") %>%
#   filter(merchant != "'M1053599405'") %>%
#   filter(merchant != "'M117188757'") %>%
#   filter(merchant != "'M1313686961'") %>%
#   filter(merchant != "'M1352454843'") %>%
#   filter(merchant != "'M1400236507'") %>%
#   filter(merchant != "'M1416436880'") %>%
#   filter(merchant != "'M1600850729'") %>%
#   filter(merchant != "'M1726401631'") %>%
#   filter(merchant != "'M1788569036'") %>%
#   filter(merchant != "'M1823072687'") %>%
#   filter(merchant != "'M1842530320'") %>%
#   filter(merchant != "'M1872033263'") %>%
#   filter(merchant != "'M1913465890'") %>%
#   filter(merchant != "'M1946091778'") %>%
#   filter(merchant != "'M348934600'") %>%
#   filter(merchant != "'M349281107'") %>%
#   filter(merchant != "'M45060432'") %>%
#   filter(merchant != "'M677738360'") %>%
#   filter(merchant != "'M85975013'") %>%
#   filter(merchant != "'M97925176'")

# ################## This would be working towards increasing prediction power
# # only in the current dataset. When new observation are being added in a 
# # system, this could lead to problems, as some useful observation could be erased.
# # In the end, it is genuinely, selective undersampling.
# customer_fraud_freq <- plyr::count(bankSim_filter, c("customer", "fraud"))
# dupl_data <- customer_fraud_freq[duplicated(customer_fraud_freq$customer), ]
# #colnames(dupl_data) <- "customer"
# 
# #getting those customers that have exhibited fraud
# bankSim_filter_cust <- bankSim_filter %>%
#   filter(customer %in% dupl_data$customer) %>%
#   select(customer) %>%
#   distinct(customer) %>%
#   arrange(customer)
# 
# bankSim_filter_total <- bankSim_filter %>%
#   filter(customer %in% bankSim_filter_cust$customer)

###############################################################################

###### Attempt to create automatisation for removing unnecessary observations
# for (category in colnames(bankSim)){
#   filtered <- plyr::count(bankSim, c(category, "fraud"))
#   dupl_data <- filtered[duplicated(filtered[, category]), ]
#   #colnames(dupl_data) <- "customer"
# 
#   #getting those customers that have exhibited fraud
#   bankSim_filter_category <- bankSim %>%
#     filter(category %in% dupl_data[, category])  %>%
#     select(category) %>%
#     distinct(category) %>%
#     arrange(category)
# 
#   bankSim_filter_total <- bankSim %>%
#     filter(category %in% bankSim_filter_category[, category])
# }
