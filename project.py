import pandas as pd #data analysis and manipulation
import matplotlib.pyplot as plt #basic visualization
import numpy as np #array operations
import seaborn as sns #advanced visualization

#importing the dataset using absolute path
titanic_df = pd.read_csv('/home/anand/Desktop/EDA/TitanicAnalysis/train.csv')

#print(titanic_df.head()) #gets the first 5 data from the file

print(titanic_df.isnull().sum()) #represents total null values in each columns
print()

survived_count = titanic_df["Survived"].value_counts() #returns the occurence of each unique entries in the column 'Survived'
print(survived_count)

#printing no of passengers who have survived
survival = titanic_df['Survived'].sum()
print("survived passengers : ",survival)

#creating a series dataframe
#series = pd.Series([549,342,981],["Non-Survivor","Survivor","Total"])

# barchart visualization for survived_count along with total
# plt.bar(survived_count.index,survived_count.values)
# plt.xticks([0,1],["Non-Survivor","Survivor"])
# plt.xlabel("Survival")
# plt.ylabel("No of passengers")
# plt.title("Number of Survivors and Non-Survivors")
# plt.show()



#piechart for survival rate
survival_rate = titanic_df['Survived'].mean()
print("Overall Survival Rate : ",round(survival_rate*100,2),"%")

plt.pie(survived_count.values,labels=survived_count.index,autopct="%1.1f%%")
plt.title("Overall Survival Rate")
#plt.show()

print()
sex_counts = titanic_df['Sex'].value_counts()
print(sex_counts)

# plt.bar(sex_counts.index,sex_counts.values,)
# plt.xlabel("Gender")
# plt.ylabel("Survivors")
#plt.show()

#piechart using gender survival
total = len(titanic_df)
male_s = sex_counts['male']/total;
female_s = sex_counts['female']/total;
print(round(male_s*100,2))
print(round(female_s*100,2))
# plt.pie(sex_counts.values,labels=sex_counts.index,autopct="%1.1f%%")
# plt.title("Survival rate by gender")
# plt.show()


# Group the data by sex and survival
sex_survival_groups = titanic_df.groupby(['Sex', 'Survived'])

# Count the number of survivors and non-survivors by sex
sex_survival_counts = sex_survival_groups.size()

# Extract the counts of male and female survivors and non-survivors
male_counts = sex_survival_counts['male']
female_counts = sex_survival_counts['female']

male_survived = male_counts[1]
male_not_survived = male_counts[0]

female_survived = female_counts[1]
female_not_survived = female_counts[0]

# Print the results
print("Male Survivors: ", male_survived)
print("Male Non-Survivors: ", male_not_survived)
print("Female Survivors: ", female_survived)
print("Female Non-Survivors: ", female_not_survived)

# Plot a bar chart of the number of male and female survivors
# plt.bar(['Male', 'Female'], [male_survived, female_survived], color=['blue', 'pink'])
# plt.xlabel("Sex")
# plt.ylabel("Number of Passengers")
# plt.title("Number of Male and Female Survivors")
# plt.show()

# Plot a pie chart of the number of male and female survivors
# plt.pie([male_survived, female_survived], labels=['Male', 'Female'], autopct="%1.1f%%", colors=['blue', 'pink'])
# plt.title("Proportion of Male and Female Survivors")
# plt.show()


# Count the number of classes
num_classes = len(titanic_df["Pclass"].unique())


# Count the number of passengers in each class
class_counts = titanic_df["Pclass"].value_counts()

# Calculate the percentage of passengers in each class
class_percents = (class_counts / titanic_df["Pclass"].count()) * 100
# Print the results
print("Number of Classes: ", num_classes)
print("Class Counts: \n", class_counts)
print("Class Percents: \n", class_percents)

# Plot a bar chart of the number of passengers in each class
# plt.bar(class_counts.index, class_counts.values)
# plt.xlabel("Passenger Class")
# plt.ylabel("Number of Passengers")
# plt.title("Number of Passengers by Class")
# plt.show()

# Plot a pie chart of the percentage of passengers in each class
# plt.pie(class_percents.values, labels=class_percents.index, autopct="%1.1f%%")
# plt.title("Percentage of Passengers by Class")
# plt.show()



# Group the data by class and gender and calculate the count of survivors
survivors_by_class_gender = titanic_df.groupby(['Pclass', 'Sex', 'Survived']).size().reset_index(name='Count')

# Filter the data for only survivors
survivors_by_class_gender = survivors_by_class_gender[survivors_by_class_gender['Survived'] == 1]

# Pivot the data to create separate columns for male and female survivors
survivors_by_class_gender_pivot = survivors_by_class_gender.pivot(index='Pclass', columns='Sex', values='Count')

# Plot the bar chart
survivors_by_class_gender_pivot.plot(kind='bar', rot=0)
# plt.title('Number of Survivors by Class and Gender')
# plt.xlabel('Class')
# plt.ylabel('Number of Survivors')
# plt.legend(title='Gender')
# plt.show()

# Plot the pie chart for female survivors
female_survivors = survivors_by_class_gender[survivors_by_class_gender['Sex'] == 'female']
# plt.pie(female_survivors['Count'], labels=female_survivors['Sex'], autopct='%1.1f%%')
# plt.title('Female Survivors')
# plt.show()

# Plot the pie chart for male survivors
male_survivors = survivors_by_class_gender[survivors_by_class_gender['Sex'] == 'male']
# plt.pie(male_survivors['Count'], labels=male_survivors['Sex'], autopct='%1.1f%%')
# plt.title('Male Survivors')
# plt.show()



embarked_counts = titanic_df["Embarked"].value_counts()
print(embarked_counts)

# sns.countplot(x="Embarked", data=titanic_df)
# plt.title("Passengers embarked from each port")
# plt.show()

survived_by_embarked = titanic_df.groupby("Embarked")["Survived"].mean()
print(survived_by_embarked)

sns.barplot(x="Embarked", y="Survived", data=titanic_df)
# plt.title("Survival rate by port of embarkation")
# plt.show()


# Find rows for Jack and Rose
jack = titanic_df.loc[(titanic_df['Name'].str.contains('Jack')) & (titanic_df['Sex'] == 'male')]
rose = titanic_df.loc[(titanic_df['Name'].str.contains('Rose')) & (titanic_df['Sex'] == 'female')]

# Print the rows for Jack and Rose
print("Jack's row:")
print(jack)

print("\nRose's row:")
print(rose)
#Here, we are using the loc method to select rows from the titanic_df DataFrame that meet certain criteria. Specifically, we are looking for rows where the Name column contains the string "Jack" and where the Sex column is "male" (for Jack's row), and where the Name column contains the string "Rose" and where the Sex column is "female" (for Rose's row).

#Once we have found the rows for Jack and Rose, we print them out using the print function.
