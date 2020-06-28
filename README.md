# PD-validation-metric

**Files Structure**

Flask will try to find the HTML files in the templates folder, in the same folder in which the python script is present.

Kindly follow the below file structure:

<pre>•	Application Folder (where python script is saved)

              o	PyCode.py

              o	templates

                    	files.html

                    	ECB.html</pre>



**Steps to Execute the tool**
<pre>
1) Change the column names of the input files as follows:

      a) Snapshots.xlsx
          i)	Unique Customer ID column = Customer ID
          ii)	Exposure column = Exposure
          iii)	Default Flag column=Default flag
          iv)	Model CRR column = Model CRR
           v)	Final CRR column = Final CRR
          vi)	Model ID column = Model ID
          vii)	Snapshot date column = Snapshot Date
          viii) Override Category Code column= Override Category Code


       b) Grades info.xlsx 
          i)	Grades column = Possible Grades
          ii)	Grades flag column = Grades Flag
          iii)	Order of Grades column (for mapping) = Ranking
          iv)	PD scale = Mid PD

       
       c) Initial Development info.xlsx
          i)	Model ID column = Model Name
          ii)	Area under the curve column = AUC
          iii)	Coef. of variation column = Coefficient of variation
          iv)	Herfindahl index column = Herfindahl Index

2) Open the show_ECB.py in any python IDE, preferably Spyder.

3) Execute the python code (for Spyder - press F5).

   A message in Python shell informs you that

	* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

4) Open the above URL (localhost:5000) in the browser. First page of the tool will be displayed on the browser.

5) On the first page you will see the file uploads option. Upload all the files as specified (Files should be in .xlsx format).

6) Click on Next Button and you will see the different models and snapshots on which we can calculate all the metrics.

7) Select Model(s) and Snapshots(s) and the desired metric you want to compute.

8) Click on Search Button and you will be able to see the results as per the selected choices.</pre>
