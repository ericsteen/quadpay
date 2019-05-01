import featuretools as ft
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,8)

orders = pd.read_csv('orders.csv')
print(orders.isnull().sum())
# Missing Values:
# customer_billing_zip           11
# customer_shipping_zip          23
# paid_installment_1              2
# paid_installment_2              0
# paid_installment_3              4
# paid_installment_4            577

# sns.heatmap(orders.isnull())

# These columns are not useful for our purposes
drop_cols = ['approved_for_installments', 'customer_shipping_zip', 'customer_billing_zip']
filtered_orders = orders.drop(drop_cols,axis=1)

# fill missing values as 'benefit of doubt' paid for our purposes
filtered_orders['paid_installment_1'] = filtered_orders['paid_installment_1'].fillna(1.0)
filtered_orders['paid_installment_3'] = filtered_orders['paid_installment_3'].fillna(1.0)

# add 'defaulted' column and make True if paid_installment_1-4 are Not Null and unpaid
def label_defaulted(row):
    res = False
    for i in range(1,5):
        # only take into account paid_installment_4 if it is not NaN
        if row['paid_installment_{}'.format(i)] == 0.0:
          res = True
    return res

filtered_orders['defaulted'] = filtered_orders.apply(lambda row: label_defaulted(row), axis=1)

merchant_count = filtered_orders['merchant_id'].value_counts()
sns.set(style="darkgrid")
sns.barplot(merchant_count.index, merchant_count.values, alpha=0.9)
plt.title('Frequency Distribution of Merchants')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Merchant', fontsize=12)
plt.show()
# print(filtered_orders['merchant_id'].value_counts())

## dummy vars for categoricals
order_id = pd.get_dummies(filtered_orders['order_id'],drop_first=True)
customer_id = pd.get_dummies(filtered_orders['customer_id'],drop_first=True)
merchant_id = pd.get_dummies(filtered_orders['merchant_id'],drop_first=True)
checkout = pd.get_dummies(filtered_orders['checkout_started_at'],drop_first=True)
decision = pd.get_dummies(filtered_orders['credit_decision_started_at'],drop_first=True)
filtered_orders.drop(['order_id', 'customer_id', 'merchant_id', 'checkout_started_at', 'credit_decision_started_at'],axis=1,inplace=True)
filtered_orders = pd.concat([filtered_orders, order_id, customer_id, merchant_id, checkout, decision],axis=1)
filtered_orders.info()

# filtered_orders.to_csv("processed_data/cleaned_orders.csv",index=False)

# Summarize and Visualize data for exploratory analysis

fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='defaulted', data=filtered_orders,ax=axs[0])
axs[0].set_title("Frequency of Default")
filtered_orders.defaulted.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Defaulted Percentage")
plt.show()

plt.figure(figsize=(12, 7))
sns.boxplot(x='defaulted',y='customer_age',data=filtered_orders,palette='winter')
plt.show()

plt.figure(figsize=(12, 7))
sns.boxplot(x='defaulted',y='customer_credit_score',data=filtered_orders,palette='winter')
plt.show()

plt.figure(figsize=(12, 7))
sns.boxplot(x='defaulted',y='order_amount',data=filtered_orders,palette='winter')
plt.show()


# use 'defaulted' as target column for modeling and see if the other factors
# can predict the default

from sklearn import metrics

filtered_orders = filtered_orders.drop('paid_installment_4',axis=1)
X_train, X_test, y_train, y_test = train_test_split(filtered_orders.drop('defaulted',axis=1),
                                                    filtered_orders['defaulted'], test_size=0.30,
                                                    random_state=101)
# Train
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Evaluate
# We can check precision, recall,f1-score using classification report

predictions = logmodel.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

cnf_matrix = metrics.confusion_matrix(y_test, predictions)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
