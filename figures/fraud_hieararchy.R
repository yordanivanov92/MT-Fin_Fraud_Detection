library(data.tree)
fraud_category <- Node$new("Financial Fraud")
  man_fraud <- fraud_category$AddChild("Management Fraud")
    fin_statement <- man_fraud$AddChild("Financial Statement Fraud")
    cust_fraud <- fraud_category$AddChild("Customer Fraud")
      cred_card <- cust_fraud$AddChild("Credit Card Fraud")
        cred_card_type1 <- cred_card$AddChild("Application Fraud")
        cred_card_type2 <- cred_card$AddChild("Behavioural Fraud")
           behav1 <- cred_card_type2$AddChild("Stolen/Lost Card")
           behav2 <- cred_card_type2$AddChild("Counterfeit Card")
           behav3 <- cred_card_type2$AddChild("Card Not Present")
           behav4 <- cred_card_type2$AddChild("Mail Theft")
      insurance <- cust_fraud$AddChild("Insurance Fraud")
SetGraphStyle(fraud_category, rankdir = "TB")
SetEdgeStyle(fraud_category, arrowhead = "vee", color = "grey35", penwidth = 2)
SetNodeStyle(fraud_category, style = "filled,rounded", shape = "box", fillcolor = "GreenYellow",
             fontname = "helvetica", tooltip = GetDefaultTooltip)
plot(fraud_category)