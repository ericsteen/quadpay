## QuadPay Machine Learning Engineer Assignment
#### As a Machine Learning Engineer at QuadPay you’ll help solve interesting engineering problems on a daily basis, including engineering the systems that generate the machine learning models for fraud prevention and real-time credit-decisioning. These models will power a platform that is processing millions of transactions every month across the largest e-commerce brands in the USA.


#### For this assignment, you’ll be supplied with an anonymized dataset of historical orders and their repayment status.


#### You’ll do some basic feature engineering and data analysis on the data to demonstrate your data, engineering, and communication skills and hopefully your ability to learn new things quickly if this isn’t all familiar to you already.  As a bonus, you may also demonstrate your skills with machine learning and/or data science.  This is not required and you should be careful not to sacrifice demonstrating your data, engineering, and communications skills in pursuit of this bonus.


#### We expect you to spend around 4 hours on this assignment, and to treat it like a day-in-the-life at QuadPay. You’re free to use the Python tools and libraries you are familiar with and whichever resources that will help you get the job done.  If you are not familiar with the Pandas or Seaborn libraries, you may find them useful in completing this assignment and we do utilize both of them.

## Data Challenge

#### QuadPay is a payment gateway that lets consumers split purchases into 4 interest free installments, every two weeks. The first 25% is taken when the order is received, and the remaining 3 installments of 25% are automatically taken every 14 days. We help customers manage their cash-flow and we help merchants increase conversion rates and average order values.


#### It may help to see our product in action at one of our merchant’s sites.


#### This assignment is designed to help you become familiar with our problem domain and start to think about which scenarios we should anticipate going forward. It gives us an opportunity to evaluate how you approach complex problems.

## Training data
#### Orders.csv contains an anonymized set of customer orders, labelled with details about which installments the customer paid. It has the following columns:

- order_id : String

- customer_id : String

- merchant_id : String

- order_amount : Decimal

- checkout_started_at : Datetime

- credit_decision_started_at : Datetime

- approved_for_installments: Boolean

- customer_credit_score: Integer

- customer_age : Integer

- customer_billing_zip : String

- customer_shipping_zip : String

- paid_installment_1 : Boolean

- paid_installment_2 : Boolean

- paid_installment_3 : Boolean

- paid_installment_4 : Boolean


###### Note that customers may have multiple orders in this dataset.

## Objective
#### An order is considered as defaulted if any of the installments have not been paid. Our aim is to keep approval rates high (allow as many customers to transact as possible) while reducing the total defaulted payments.


#### Your first task is to ingest the data and transform into features suitable for training machine learning models.  Very briefly and non-exhaustively, machine learning models only take in rows (arrays) of numbers as inputs for training.  The columns are referred to as features.  So strings, dates, times, numbers that are actually ids (e.g. enum values, guids, hashes, zip codes, etc.), and so on need to be transformed into numbers or dropped.  Such transformation is a subset of feature engineering.  You may either manually code it or use a library that tries to automate the process such as the featuretools library.  Note that while featuretools looks useful, we have not actually tried and can not vouch for it (i.e. use at your own risk).


#### You second task is to provide to a visual analysis.  The analysis can be very simple and it may done directly on the features you just created in the first task.  The primary point of this task is to demonstrate your ability to create visualizations from data using Python code (i.e. not data visualization products  like Power BI or Tableau).  You may also provide visual analysis of the predictions produced by a machine learning you’ve trained or the kinds of visual analysis typically done at this stage of a data science project.  But again it is not necessary to apply data science or train machine learning models and doing so is purely a bonus.


## Please submit a code repository with instructions for how to execute your tool(s).


## Questions you may with to answer:

- What did you learn from this exercise?

- What do you think should get bonus points for?

## Submission
#### Please send us your final code repository. We’ll schedule a follow-up call with some of our engineers to discuss your solution.
