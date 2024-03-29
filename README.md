# AI Academy Capstone Project Team 2
## Repo Contents
* Download the dataset from [here](https://www.kaggle.com/datasets/willianoliveiragibin/websites-e-comerce) and place it in the folder with this readme.
* images - folder containg all the images within the README
* [Capstone Proposal.pdf](https://github.com/s-callejo/AI-Academy-capstone-project-team2/blob/main/Capstone%20Proposal.pdf) - File containing capstone proposal details
* [README.md](https://github.com/s-callejo/AI-Academy-capstone-project-team2/blob/main/README.md) - A description of the project goals, process, and results
* [eda.ipynb](https://github.com/s-callejo/AI-Academy-capstone-project-team2/blob/main/eda.ipynb) - Jupyter Notebook containing Exploratory Data analysis
* [index.ipynb](https://github.com/s-callejo/AI-Academy-capstone-project-team2/blob/main/index.ipynb) - Main Jupyter Notebook containing the models and anaylisis for the project
* [Capstone Project.pdf](/Capstone%20Project.pdf) - The presentation for this project.

## Required Packages
* pandas
* sklearn
* imblearn
* collections
* matplotlib
* seaborn

In the case of imblearn package not working, reinstall sklearn and imblearn using the commands bellow.
> pip uninstall scikit-learn --yes <br>
> pip uninstall imblearn --yes <br>
> pip install scikit-learn==1.2.2 <br>
> pip install imblearn

## Goal

In the dynamic realm of e-commerce, this project aims to address a critical challenge that is understanding the factors that drive customer returns, with a focus on enhancing profitability. Conventional approaches focus on maximizing sales, but our unique proposition is to unravel the complexities of dissatisfaction leading to product returns. By analyzing the feature importance within the dataset, we intend to provide actionable recommendations for businesses to minimize returns, thereby fostering relationships with customers. 

## Data

The raw dataset has 172838 entries with 15 columns that include: 'accessed_date', 'duration_(secs)', 'network_protocol', 'ip', 'bytes', 'accessed_Ffom', 'age', 'gender', 'country', 'membership', 'language','sales', 'returned_amount', 'pay_method' with target variable being 'returned'. This dataset was taken from the time frame of a week.

![](images/data_before.PNG)

## Data Pre-Processing and Exploration

We decided to drop 'network_protocol', 'ip', 'bytes', 'sales', and 'returned_amount'.

![](/images/data_after.PNG)

### Age
When looking into the age column, we saw that there were placeholder values that represented null items. There were 0s and dashed lines. We removed those values as it was a way not to skew the data because replacing the mean, median, or mode in those values would make an our model more biased to that value in the age feature.

![](/images/age.PNG) 

### Gender
The process of removing the placeholder values from the Age column also removed the Unknown values from the Gender column.

![](/images/gender.PNG)

### Payment Methods
We have 4 payment methods. No need to deal with null, duplicates, or outliers

![](/images/payment_methods.PNG)

### Membership Type
When the 0s and null value Age rows were removed, the Not Logged In membership class is removed, leaving only the normal and premium classes for the model. No need to deal with null, duplicates, or outliers.

![](/images/membership_type.PNG)

### Languages
There are 29 different languages used on the e-commerce site. The top 3 most common were English Chinese and Spanish. The format is not consistent across the column so we capitalized the first letter of each language to provide consistency. There are no null values to address or any outliers.

![](/images/language.PNG)

### Access Methods
There are 7 different access methods found in the database; Android App, Chrome, Mozilla Firefox, iOS App, Safari, Microsoft Edge and Others. SafFRi is a misspelling of Safari that was fixed before modeling. There are no null values to address or any outliers.

![](/images/access_method.PNG)

### Countries
There are 27 countries, all displayed as country codes, that users accessed the e-commerce site from. The top 3 were Italy, United States and Canada. No need to deal with null, duplicates, or outliers.

![](/images/country.PNG)

### Return (Target Variable)
The thing that needs to be addressed with the target variable is the idea of an unbalanced data set. There is only about 15% of the data that contains customers that returned an item.

![](/images/target_variable.PNG)

## Returned Items Only

We also looked at the distributions for each column where the returned value is true. We found that the distributions in each column in the overall dataset and subset where returned is true are exteremely similar.

### Age

![](/images/r_age.PNG)

### Gender

![](/images/r_gender.PNG)

### Payment Method

![](/images/r_payment_methods.PNG)

### Membership Type

![](/images/r_membership_types.PNG)

### Languages

![](/images/r_language.PNG)

### Access Methods

![](/images/r_access_method.PNG)

### Countries

![](/images/r_country.PNG)

## Imbalanced Data and Solutions

As mentioned earlier, our target variable being imbalanced brings a few problems; possibility of bias toward the majority class, resulting in poor performance on the minority class, metrics such as accuracy can be misleading as high accuracy can be achieved by predicting the majority class, and the model may focus more on features that help distinguish the majority class, neglecting features important for minority class prediction.

![](/images/returned_nonreturned_barchart.png)

To resolve this problem we used Oversampling/Undersampling and SMOTE. 

### Oversampling
Oversampling duplicates the minority class until there are an equal number of records in the minority class as majority class. 

![](/images/oversampling.PNG)

### Undersampling

Undersampling takes pieces of the majority class to reduce the used portion of the majority class to have the same number of records as the minority class. 

![](/images/undersampling.PNG)

### SMOTE

SMOTE generates new records in the minority class to increase the number of records to equal the number in the majority class.

![](/images/SMOTE.PNG)

## Modeling

In this project we used Logistic Regression and Random Forest models. We conducted a grid search of hyperparameters to tune basic models of each in an attempt to achieve better models.

For logistic regression, we decided to change the solver and C (penalty strength) hyperparameters. With how our baseline model of logistic regression was performing, we needed to edit highly impactful hyperparameters to change how the logistic regression model was learning. One step was to change the solver. The solver is the type of algorithm used for optimization. The thought behind changing the solver was to see if the model could find a different way optimize the data, then maybe it could give us better accuracy. In addition to the solver, the C hyperparameter was also changed for overfitting reasons. It controls the strength of regularization in which it could make the model either simple or complex.

For random forest, the hyperparameters max_depth, and n_estimators help limit overfitting of a model. They both have a type of cutoff point wherein the model will stop training at where we set the cutoff point and then it stops the model from training too much which would cause overfitting. As we can see with our accuracy scores, the model tend to overfit in the training data so it seems best to edit these hyperparameters to try and get a better model.


## Results

For logistic regression, changing the hyperparameters made the oversampled set of training data have almost no change in accuracy rate (from .54 to .5405 for training and .3988 to .3988 for test). As for the under sample, there were only slight improvements (from .5389 to .5434 for the training and from .401 to .4026 for the test). For the SMOTE oversampling method, the change made the model slightly better for the training and the testing.

For random forest, the hyperparameters found by grid search made the model performance worse on the unseen data. When using the over sampled set, the training accuracy went from .9586 to .9494 and the test accuracy went from .742 to .726. When using the under sampled set, the training accuracy went from .9616 to .6265 and the test accuracy went from .5149 to .3744. For the SMOTE oversampling, the same can be seen in which the model become worse when the new hyperparameters are applied.

In the ROC curve, we can see that the training is overfitting and the test set is only ever so slightly better than random chance.

![](/images/ROC_Curve.png)

In the confusion matrix, it can contribute to the idea the model is being biased to predicting non-returned items as proposed by our imblanced data problem.

![](/images/confusion_matrix.png)

The precion and recall table also contrinbutes to this idea as the proportions of predicting 0 (non-returned) as better than 1 (returned)

![](/images/precision_recall_table.png)

## Conclusion

In conclusion, our exploration into the ecommerce dataset had shed light on critical insights and areas of improvement. First, we showcased the impact of attributes such as age, membership type, language, and gender on the likelihood of returns. For us to enhance our findings, we recognize the need for richer reviews and additional data collection. Elevating the predictive power of our model can be achieved by using neural network tailored to our dataset and introducing more detailed information on the reasons for returns. The next steps involve gathering more diverse and detailed data, embracing customer feedback to inform marketing strategies, and refining the returns process to bolster customer satisfaction and, consequently, profitability. 

Given these next steps, having more data may allow us to predict whether or not an item is returned, in which that model could give insight on which features would be best to focus on that have the greatest impact on a returned item. The features that are chosen could be used for developing a marketing strategy against those features to be able to lessen that amount of returns. In addition to marketing strategies, when our model is able to predict whether an item is returned or not based on these features, we can use those predictions as a way for budgeting on a possibly monthly or quarterly basis.
