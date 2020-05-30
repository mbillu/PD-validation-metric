# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:55:15 2020

@author: Mohd Bilal
"""
    
import pandas as pd
import numpy as np
import math
from scipy.stats import norm    
from scipy.stats import mannwhitneyu
from scipy.stats import beta
from statistics import variance


pd.options.mode.chained_assignment = None

#Import the snapshots file, grades file and initial developement calculations file
#prior_df=pd.read_excel("Snapshots.xlsx",sheet_name="PRIOR")
#current_df=pd.read_excel("Snapshots.xlsx",sheet_name="CURRENT")
#grades_df=pd.read_excel("Grades info.xlsx",sheet_name="Internal_Ratings")
#initial_data_df=pd.read_excel("Initial Development info.xlsx",sheet_name="Information")

C_id='Customer ID'
Grades="Possible Grades"
expo="Exposure"
Flag="Grades Flag"         #from Grades info.xlsx
def_flag="Default flag"
mCRR="Model CRR"
fCRR="Final CRR"



#calculates row wise sum of active customers
def Ni_cal(mat,M,N,a):
    
    if a==1:
        return mat.iloc[:M,:N].sum(axis=1).tolist()
    else:
        return mat.iloc[:M,:N].sum(axis=0).tolist()


def mig_mat_calculate(sub_prior,sub_current,CRR): 
    temp_list=[]
    
    #mapping grades from external file
    grades_row=grades_df[grades_df[Flag]==1][Grades].tolist()
    grades_column=grades_df[Grades].tolist()
    
    
    
    #calculate migrations
    for i in range(0,len(grades_row)):
        cust_with_grade_i = sub_prior[sub_prior[CRR]==grades_row[i]][[C_id]]
        
        cust_in_current_df = cust_with_grade_i.merge(sub_current[[C_id,CRR]], on=C_id, how="inner")
        
        temp_list.append((cust_in_current_df.groupby([CRR]).size()))
        
        #print(cust_in_current_df.head())

    rows,cols=(len(grades_row),len(grades_column))
    
    migration_mat=[[0 for j in range(cols)] for i in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if grades_column[j] in  temp_list[i].index.tolist():
                migration_mat[i][j]=temp_list[i][grades_column[j]]
    

    return (migration_mat,grades_row,grades_column)
    
    
#converts 2d matrix into dataframe(only for visulization purpose)
def mat_display(migration_mat,grades_prior,grades_current):

    mig_dict={}
    j=0
    
    for i in migration_mat:
        mig_dict[grades_prior[j]]=i[0:]
        j+=1
    
    df1=pd.DataFrame(mig_dict,index=grades_current)
    df2=df1.transpose()
    
    #df2["Dropped Customers"]=temp_list
    
    df2.loc["Total",:]=df2.sum(axis=0)
    df2.loc[:,"Total"]=df2.sum(axis=1)
    
    df2[df2.eq(0)]=np.nan

    #print("\nMigration matrix\n",df2)    
    #df2.to_csv("Migration Matrix.csv")   #write to csv fie
    
    return df2
    


#1st statbilty metric calculation(As per ECB)
def Customer_Migrations(migration_mat,mat_df):
    
    MNU=0
    MNL=0
    
    Ni_series = Ni_cal(mat_df,k,k,1)   # for row-wise sum pass 1, for column-wise sum pass 0
    
    for i in range(1,k):
        pij_sum=0
        for j in range(i+1,k+1):
            if Ni_series[i-1]!=0:
                pij_sum=pij_sum+(migration_mat[i-1][j-1]/Ni_series[i-1])
                
        if Ni_series[i-1]!=0:
            MNU=MNU+(max(abs(i-k),abs(i-1)) * Ni_series[i-1] * pij_sum)

    for i in range(2,k+1):
        pij_sum=0
        for j in range(1,i-1):
            if Ni_series[i-1]!=0:
                pij_sum=pij_sum+(migration_mat[i-1][j-1]/Ni_series[i-1])
        
        if Ni_series[i-1]!=0:
            MNL=MNL+(max(abs(i-k),abs(i-1)) * Ni_series[i-1] * pij_sum)
    
    
    U_MWB=0
    L_MWB=0

    for i in range(1,k):
        for j in range(i+1,k+1):
            if Ni_series[i-1]!=0:
                U_MWB=U_MWB+abs(i-j)*Ni_series[i-1]*(migration_mat[i-1][j-1]/Ni_series[i-1])
            
    try:
        U_MWB=U_MWB/MNU
    except ZeroDivisionError:
        print("No Downgrade")
                
    for i in range(2,k+1):
        for j in range(1,i-1):
            if Ni_series[i-1]!=0:
                L_MWB=L_MWB+abs(i-j)*Ni_series[i-1]*(migration_mat[i-1][j-1]/Ni_series[i-1])
            
    try:
        L_MWB=L_MWB/MNL
    except ZeroDivisionError:
        print("No Upgrade")

    #print("\nUpper MWB = ",U_MWB,"\nLower MWB = ",L_MWB)
    
    data={"Upper MWB":[U_MWB],"Lower MWB":[L_MWB]}
    
    df=pd.DataFrame(data)
    
    #df.to_csv("Cutomer Migrations.csv",index=False)   #write to csv fie
    
    return df
    


#2nd statbilty metric calculation(As per ECB)
def stability_of_migration_matrix(grades_prior,grades_current,migration_mat,mat_df):
    
    Ni_series = Ni_cal(mat_df,k,k,1)   # for row-wise sum pass 1, for column-wise sum pass 0
    
    rows,cols=(k,k)

    z_mat=[[np.nan for j in range(cols)] for i in range(rows)]
    
    np.seterr(all='raise')
    
    for i in range(rows):
        for j in range(cols):
            if Ni_series[i]!=0:
                p_ij=migration_mat[i][j]/Ni_series[i]
            else:
                continue
            
            if j<i:
                if Ni_series[i]!=0:
                    p_ij1=migration_mat[i][j+1]/Ni_series[i]
                else:
                    continue
                        
                if (p_ij1-p_ij)==0:
                        z_mat[i][j]=0
                else:
                    try:
                        z_mat[i][j]=round((p_ij1-p_ij)/math.sqrt(((p_ij*(1-p_ij))/Ni_series[i]) + ((p_ij1*(1-p_ij1))/Ni_series[i]) + ((2*p_ij*p_ij1)/Ni_series[i])),3)
                    except (FloatingPointError,ValueError,ZeroDivisionError):
                        z_mat[i][j]=np.nan
                        continue
                    
            elif j>i:
                if Ni_series[i]!=0:
                    p_ij2=migration_mat[i][j-1]/Ni_series[i]
                else:
                    continue
                        
                if (p_ij2-p_ij)==0:
                        z_mat[i][j]=0
                else:
                    try:
                        z_mat[i][j]=round((p_ij2-p_ij)/math.sqrt(((p_ij*(1-p_ij))/Ni_series[i]) + ((p_ij2*(1-p_ij2))/Ni_series[i]) + ((2*p_ij*p_ij2)/Ni_series[i])),3)
                    except (FloatingPointError,ValueError,ZeroDivisionError):
                        z_mat[i][j]=np.nan
                        continue

    phi_mat=[[np.nan for j in range(cols)] for i in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if i!=j and not(math.isnan(z_mat[i][j])):
                phi_mat[i][j]=round(norm.cdf(z_mat[i][j],loc=0,scale=1),3)
    
    j=0
    mig_dict={}
    
    for i in z_mat:
        mig_dict[grades_prior[j]]=i[0:]
        j+=1
    
    df=pd.DataFrame(mig_dict,index=grades_current[:-l])
    z_mat_df=df.transpose()
    
    #print("\nZij\n",z_mat_df)
    #z_mat_df.to_csv("Stability of migration matrix_Zij.csv")   #write to csv fie
    
    
    j=0
    mig_dict={}
    
    for i in phi_mat:
        mig_dict[grades_prior[j]]=i[0:]
        j+=1
        
    df1=pd.DataFrame(mig_dict,index=grades_current[:-l])
    phi_mat_df=df1.transpose()

    #print("\nPHIij\n",phi_mat_df)
    #phi_mat_df.to_csv("Stability of migration matrix_PHIij.csv")   #write to csv fie
    
    return z_mat_df,phi_mat_df



#3rd statbilty metric calculation(As per ECB)
def concentration_in_RG(sub_current,matrix_df,CV_init):
    
    Ni_series = Ni_cal(matrix_df,k,k,1)   # for row-wise sum pass 1, for column-wise sum pass 0
    
    cust_sum=sum(Ni_series)
    
    temp_sum=0

    for i in range(k):
        if Ni_series[i]!=0:
            temp_sum=temp_sum + ((Ni_series[i]/cust_sum) - 1/k) * ((Ni_series[i]/cust_sum) - 1/k)
    
    
    CV_curr=math.sqrt(k*temp_sum)
    
    HI_curr=1+(math.log(((CV_curr * CV_curr)+1)/k)/math.log(k))
    
    p_val=round(norm.cdf((math.sqrt(k-1)*(CV_curr-CV_init))/(math.sqrt(CV_curr*CV_curr*(0.5+(CV_curr*CV_curr))))),4)
    
    #print("\nNumber Weighted\nCVcurr = ",CV_curr,"\nHIcurr = ",HI_curr,"\np-value = ",p_val)
    
    last_column=matrix_df["Total"]
    
    last_column.fillna(0,inplace=True)
    
    #print(sub_current.groupby([fCRR])[expo].agg('sum'),last_column)
    expo_list=sub_current.groupby([fCRR])[expo].agg('sum').tolist()
    
    if len(expo_list)==(k+l):
        expo_list=expo_list[:-l]
    
    expo_sum=sum(expo_list)
    
    temp_sum=0
    j=0
    
    for i in range(k):
        try:
            if last_column[i]!=0:
                temp_sum=temp_sum + ((expo_list[j]/expo_sum) - 1/k) * ((expo_list[j]/expo_sum) - 1/k)
                j+=1
        except Exception as e:
            print("Exception : ",e)
    
    CV_curr_expo=math.sqrt(k*temp_sum)
    
    HI_curr_expo=1+(math.log(((CV_curr_expo * CV_curr_expo)+1)/k)/math.log(k))
    
    p_val_expo=round(norm.cdf((math.sqrt(k-1)*(CV_curr_expo-CV_init))/(math.sqrt(CV_curr_expo*CV_curr_expo*(0.5+(CV_curr_expo*CV_curr_expo))))),4)
    
    #print("\nExposure Weighted\nCVcurr = ",CV_curr_expo,"\nHIcurr = ",HI_curr_expo,"\np-value = ",p_val_expo)

    
    
    data={"Coefficient of Variation (curr)":[CV_curr],"Herfindahl Index (curr)":[HI_curr],"p-value":[p_val]}
    
    df1=pd.DataFrame(data)
    
    #df1.to_csv("Concentration in RG no.csv",index=False)   #write to csv fie
    
    
    data1={"Coefficient of Variation (curr)":[CV_curr_expo],"Herfindahl Index (curr)":[HI_curr_expo],"p-value":[p_val_expo]}
    
    df2=pd.DataFrame(data1)
    
    #df2.to_csv("Concentration in RG expo.csv",index=False)   #write to csv fie
    
    return df1,df2



def discriminatory_pow(sub_prior,sub_current,AUC_init):
    
    common_cust_df = sub_prior[[C_id,def_flag,fCRR]].merge(sub_current[[C_id]], on=C_id, how="inner")
    
    def_cust_df=common_cust_df[common_cust_df[def_flag]==1][[C_id,fCRR]]
    non_def_cust_df=common_cust_df[common_cust_df[def_flag]==0][[C_id,fCRR]]
    
    #checking others customer 
    temp_df=non_def_cust_df.merge(sub_current[[C_id,fCRR]],on=C_id,how='inner')
    temp_df1=temp_df[temp_df[fCRR+"_y"].isin(non_def_grades)]
    
    if not temp_df1.empty:
        non_def_cust_df=temp_df1
        
        non_def_cust_df=non_def_cust_df[[fCRR+"_x"]].merge(grades_df[[Grades,'Ranking']], left_on=fCRR+"_x", right_on=Grades)
    else:
        non_def_cust_df=non_def_cust_df.merge(grades_df[[Grades,'Ranking']], left_on=fCRR, right_on=Grades)
    
    
    def_cust_df=def_cust_df[[fCRR]].merge(grades_df[[Grades,'Ranking']], left_on=fCRR, right_on=Grades)
    
    
    Data1=def_cust_df['Ranking'].tolist()
    
    Data2=non_def_cust_df['Ranking'].tolist()
    
    U = mannwhitneyu(Data1, Data2)[0]
    
    AUC = U/(len(Data1) * len(Data2))
    
    V10=[]
    V01=[]
    
    temp=0
    
    for i in Data1:
        for j in Data2:
            if i<j:
                temp+=1
            elif i==j:
                temp+=0.5
        
        V10.append(temp/len(Data2))
        temp=0;
        
    #print(V10)
    
    temp=0
    for i in Data2:
        for j in Data1:
            if i<j:
                temp+=1
            elif i==j:
                temp+=0.5
        
        V01.append(temp/len(Data1))
        temp=0;
    
    #print(V01)
    
    est_variance=(variance(V10)/len(Data1))+(variance(V01)/len(Data2))
    est_std_dev=math.sqrt(est_variance)
    
    S=(AUC_init-AUC)/est_std_dev
    
    data={"AUC (at initial validation/developement)":[AUC_init],"Current AUC":[AUC],"Test Statistics (S)":[S]}
    
    df1=pd.DataFrame(data)
    
    #print("\nAUC = ",AUC,"\nS = ",S)
    
    #df1.to_csv("AUC.csv",index=False)   #write to csv fie
    
    return df1
    
    

def pred_ability(sub_prior,sub_current,grades_row):
    
    expo_list=sub_prior.groupby([fCRR])[expo].agg('sum').tolist()

    temp_list=[]
    for i in range(0,len(grades_row)):
        cust_with_grade_i = sub_prior[sub_prior[fCRR]==grades_row[i]][[C_id]]
        
        cust_in_current_df = cust_with_grade_i.merge(sub_current[[C_id,fCRR]], on=C_id, how="inner")
        
        temp_list.append((cust_in_current_df.groupby([fCRR]).size()))
        
        
    rows,cols=(len(grades_row),2)
    
    count_mat=[[0 for j in range(cols)] for i in range(rows)]
    n,d=0,0
    for i in range(rows):
        for j in temp_list[i].index.tolist():
            if j in def_grades:
                d+=temp_list[i][j]
            if j in non_def_grades:
                n+=temp_list[i][j]
                
            count_mat[i][1]=d
            count_mat[i][0]=n+d
            
        d=0
        n=0
    
    j=0
    data={}
    
    for i in count_mat:
        N=i[1]+0.5
        D=i[0]-i[1]+0.5        
        
        data[grades_row[j]]=[grades_row[j],PD_list[j],round(i[0]),int(i[1]),round(beta.pdf(PD_list[j],N,D),4),round(norm.cdf(beta.pdf(PD_list[j],N,D)),4),expo_list[j]]
        j+=1
    
    df1=pd.DataFrame(data,index=["Grades","PD","No. of Customers","Defaulted Customers","Beta Value","p-value","Exposure"])
    df2=df1.transpose()
    
    common_cust_df = sub_prior[[C_id,fCRR]].merge(sub_current[[C_id,fCRR]], on=C_id, how="inner")
    
    common_cust_df=common_cust_df[~common_cust_df[fCRR+"_y"].isin(other_grades_df)]
    def_cust_df=common_cust_df[common_cust_df[fCRR+"_y"].isin(def_grades)][[C_id]]
    #non_def_cust_df=common_cust_df[common_cust_df[fCRR+"_y"].isin(non_def_grades)][[C_id]]
    
    pd_cust_df=common_cust_df[[fCRR+"_x"]].merge(grades_df[[Grades,'Mid PD']], left_on=fCRR+"_x", right_on=Grades)
    
    N=len(def_cust_df)+0.5
    D=len(pd_cust_df)-len(def_cust_df)+0.5
    
    PD_portfolio=sum(pd_cust_df['Mid PD'])/len(pd_cust_df)
    
    data={"Grades":"ALL","PD":[PD_portfolio],"No. of Customers":[len(pd_cust_df)],"Defaulted Customers":[len(def_cust_df)],"Beta Value":[round(beta.pdf(PD_portfolio,N,D),4)],"p-value":[round(norm.cdf(beta.pdf(PD_portfolio,N,D)),4)],"Exposure":sum(df2['Exposure'])}
    
    df3=pd.DataFrame(data)
    
    #print("\n",df2,"\n",df3)
    
    #df2.to_csv("predictive ability (grade level).csv")   #write to csv fie
    #df3.to_csv("predictive ability (portfolio level).csv")   #write to csv fie
    
    return df2,df3



def occ_of_override(sub_current):
    over_col=sub_current['Override Category Code']
    
    Summary_Stat = (len(over_col)-(over_col.isna().sum())) / len(over_col)
    
    #print("\nSummary Satistics",Summary_Stat)
    
    data={"No. of Customers":[len(over_col)],"No. of Customers with overrides":[(len(over_col)-(over_col.isna().sum()))],"Summary Statitics":[Summary_Stat]}
    
    df=pd.DataFrame(data)
    
    #df.to_csv("Override.csv",index=False)   #write to csv fie
    
    return df



from flask import Flask, render_template, request
import ctypes  # An included library with Python install.   

app = Flask(__name__)
app.debug = True

'''@app.route('/')
def student():
   return render_template('test.html')'''


@app.route('/')
def files():
   return render_template('files.html')

def abc():
    return 5+2

@app.route("/MSID",methods = ['POST', 'GET'])
def MSIDs():
    if request.method == 'POST':
        #result=request.form
        f1=request.files['myfile1']
        f2=request.files['myfile2']
        f3=request.files['myfile3']
        
        #print("******************",type(f1),"******************")
        
        global snaps_df
        global prior_df 
        global current_df 
        global grades_df 
        global initial_data_df 
        global M
        global S
        global def_grades
        global non_def_grades
        global other_grades_df
        global PD_list
        global k
        global l
        
        
        snaps_df,grades_df,initial_data_df=pd.read_excel(f1),pd.read_excel(f2),pd.read_excel(f3)

        def_grades=grades_df[grades_df[Flag]==0][Grades].tolist()          #Default grades
        non_def_grades=grades_df[grades_df[Flag]==1][Grades].tolist()      #non-defaults grades
        other_grades_df=grades_df[grades_df[Flag]==2][Grades].tolist()     #other grades
        
        initial_data_df.set_index('Model Name',inplace=True)
        
        PD_list=grades_df["Mid PD"].tolist()
        
        k=len(non_def_grades)                                       #No. of non-defaults grades
        l=len(grades_df[grades_df[Flag]!=1][Grades])   #No. of grades others then non-defaults
        #snaps_df['Snapshot Date'] = pd.to_datetime(snaps_df['Snapshot Date'])
        
        #snaps_df['Snapshot Date Parsed'] = pd.to_datetime(snaps_df['Snapshot Date'], infer_datetime_format=True)
        
        #print(snaps_df["Snapshot Date"].unique().tolist())
        M=snaps_df["Model ID"].unique().tolist()
        S_snap=snaps_df["Snapshot Date"].unique()
        S_snap=np.datetime_as_string(S_snap).tolist()
        S_snap=[i[:-19] for i in S_snap]
        
        '''S_prior=prior_df["Snapshot Date"].unique()
        S_prior=np.datetime_as_string(S_prior).tolist()
        S_prior=[i[:-19] for i in S_prior]'''
        
        S=[]
        for i in range(len(S_snap)):
            year=int(S_snap[i][:4])
            right_part=S_snap[i][4:]
            year-=1
            
            if (str(year)+right_part) in S_snap:
                S.append(S_snap[i])
        
        print("\n")
        print("Available Model(s) = ",M)
        print("Available Snapshot(s) = ",S)
        
        M.append("All Models")
        S.append("All Snapshots")
        
        
        return render_template('ECB.html',S=S,M=M)
        #print(result,"******************",abc(),"\n",dfr1,"\n",dfr2,"\n",dfr3,"\n",dfr4)
        



@app.route("/tables",methods = ['POST', 'GET'])
def show_tables():
    if request.method == 'POST':
        result=request.form
        
        M_list=[]
        S_list=[]
        
        list_of_keys=[]
        for key,value in result.items():
            if key == 'M':
                M_list=request.form.getlist('M')
                continue
            if key == 'S':
                S_list=request.form.getlist('S')
                continue
                
            list_of_keys.append(key)
        
        S_list_prior=[]
        for i in range(len(S_list)):
            if S_list[i] != 'All Snapshots':
                year=int(S_list[i][:4])
                right_part=S_list[i][4:]
                year-=1
                S_list_prior.append(str(year)+right_part)
        
        S_prior=[]
        for i in range(len(S)):
            if S[i] != 'All Snapshots':
                year=int(S[i][:4])
                right_part=S[i][4:]
                year-=1
                S_prior.append(str(year)+right_part)
        
        
        print("12-month prior available snapshot(s) = ",S_prior)
        print("Selected Model(s) = ",M_list)
        print("Selected Snapshot(s) = ",S_list)
        print("12-month prior selected snapshots = ",S_list_prior)
        print("Selected metric mapping = ",list_of_keys)
        
        if len(M_list)==0:
            ctypes.windll.user32.MessageBoxW(0, "\nPlease Select One of the option from Models.\nGo to First page(Files Upload) of the app or refresh the the page.\nThank You...", "Warning",0x1000)
        
        if len(S_list)==0:
            ctypes.windll.user32.MessageBoxW(0, "\nPlease Select One of the option from Snapshots.\nGo to First page(Files Upload) of the app or refresh the the page.\nThank You...", "Warning",0x1000)
        #dfr2['Snapshot Date']=str(dfr2.loc[0,'Snapshot Date'])[:-9]
        #print(dfr2)
        
        
        print("**********")
        if 'All Models' in M_list:
            if 'All Snapshots' in S_list:
                count=0
                result=''
                for i in range(len(M)-1):
                    for j in range(len(S)-1):
                        
                        sub_current = snaps_df[snaps_df['Model ID']==M[i]]#& (dfr2['Snapshot Date']==S_list[i])] 
                        sub_current = sub_current[sub_current['Snapshot Date']==S[j]]
                        
                        sub_prior = snaps_df[snaps_df['Model ID']==M[i]]#& (dfr2['Snapshot Date']==S_list[i])] 
                        sub_prior = sub_prior[sub_prior['Snapshot Date']==S_prior[j]]
                        
                        print("\nShape of the Snapshots Selected = ",sub_current.shape,sub_prior.shape,k,l)
            
                        
                        #calculate migration matrix in the form of list of lists (2d matrix), also calculates the possible grades in prior and current snapshot
                        migration_mat, grades_prior, grades_current = mig_mat_calculate(sub_prior,sub_current,"Final CRR") #Final CRR or Model CRR column name
                        
                        #modification in matrix into dataframe (to display)
                        migration_mat_df = mat_display(migration_mat,grades_prior,grades_current)
                        
                        #1st statbility metric calculation
                        df1=Customer_Migrations(migration_mat,migration_mat_df)
                        
                        #second stability metric calculation
                        df2, df3 = stability_of_migration_matrix(grades_prior,grades_current,migration_mat,migration_mat_df)
                        
                        #third stability metric calculation
                        CV_init=initial_data_df.loc[M[i],"Coefficient of variation"]
                        df4, df5,= concentration_in_RG(sub_current,migration_mat_df,CV_init)
                        
                        #Discriminatory Power (AUC) calculation
                        AUC_init=initial_data_df.loc[M[i],"AUC"]
                        df6 = discriminatory_pow(sub_prior,sub_current,AUC_init)
                        
                        #Predictive Ability calculation
                        df7, df8=pred_ability(sub_prior,sub_current,grades_prior)
                        
                        #Qualitative validation(Occ. of overrides) calculation
                        df9 = occ_of_override(sub_current)
                        
                        migration_mat_df.fillna('',inplace=True)
                        
                        if count==0:
                            s1='<h2>Qualitative Validation</h2><h3>Occurance of Overrides</h3>' + df9.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s2='<h2>Predictive Ability</h2><h3>PD back-testing (Portfolio level)</h3>' + df8.to_html(index=False,border=2,justify="center") + '<br>'
                            s3='<h2>Predictive Ability</h2><h3>PD back-testing (Grades level)</h3>' + df7.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s4='<h2>Discriminatory Power</h2><h3>Current AUC vs. AUC at initial validation/development</h3>' + df6.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s5='<h2>Stability</h2><h3>Migration Matrix</h3>' + migration_mat_df.to_html(border=2,justify="center") +'<br>'
                            s6='<h3>1. Customer Migrations</h3>' + df1.to_html(index=False,border=2,justify="center") + '<br>'
                            s7='<h3>2. Stability of Migration Matrix (Zij)</h3>' + df2.to_html(na_rep="",border=2,justify="center") + '<br>'
                            s8='<h3>2. Stability of Migration Matrix (PHIij)</h3>' + df3.to_html(na_rep="",border=2,justify="center") + '<br>'
                            s9='<h3>3. Concentration in Rating Grades (Number Weighted)</h3>' + df4.to_html(index=False,border=2,justify="center") + '<br>'
                            s10='<h3>3. Concentration in Rating Grades (Exposure Weighted)</h3>' + df5.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            
                            res=''
                            if 'Q1' in list_of_keys:
                                res+=s1
                            if 'P1' in list_of_keys:
                                res+=s2
                            if 'P2' in list_of_keys:
                                res+=s3
                            if 'D1' in list_of_keys:
                                res+=s4
                            if 'S1' in list_of_keys:
                                res+=s5
                            if 'S2' in list_of_keys:
                                res+=s6
                            if 'S3' in list_of_keys:
                                res+=s7
                            if 'S4' in list_of_keys:
                                res+=s8
                            if 'S5' in list_of_keys:
                                res+=s9
                            if 'S6' in list_of_keys:
                                res+=s10
                            if res=='':
                                ctypes.windll.user32.MessageBoxW(0, "\nPlease Select One of the metric.\nGo to First page(Files Upload) of the app or refresh the page.\nThank You...", "Warning",0x1000)
                            
                            print("######")
                            _file= open("templates\Result1.html", 'w')
                            _file.write('<center>' 
                                        +'<h1>Validation Results of IRB PD Model</h1><br>'
                                        +'<hr><hr><hr><hr><h1>Model ('+M[i]+') Snapshot ('+S[j]+')</h1><hr><hr><hr>'
                                        +res
                                        +'</center>')
                            _file.close()
                            result=result+'<center>'+'<h1>Validation Results of IRB PD Model</h1><br>'+'<hr><hr><hr><hr><h1>Model ('+M[i]+') Snapshot ('+S[j]+')</h1><hr><hr><hr>'+res+'</center>'
                        
                        else:
                            s1='<h2>Qualitative Validation</h2><h3>Occurance of Overrides</h3>' + df9.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s2='<h2>Predictive Ability</h2><h3>PD back-testing (Portfolio level)</h3>' + df8.to_html(index=False,border=2,justify="center") + '<br>'
                            s3='<h2>Predictive Ability</h2><h3>PD back-testing (Grades level)</h3>' + df7.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s4='<h2>Discriminatory Power</h2><h3>Current AUC vs. AUC at initial validation/development</h3>' + df6.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s5='<h2>Stability</h2><h3>Migration Matrix</h3>' + migration_mat_df.to_html(border=2,justify="center") +'<br>'
                            s6='<h3>1. Customer Migrations</h3>' + df1.to_html(index=False,border=2,justify="center") + '<br>'
                            s7='<h3>2. Stability of Migration Matrix (Zij)</h3>' + df2.to_html(na_rep="",border=2,justify="center") + '<br>'
                            s8='<h3>2. Stability of Migration Matrix (PHIij)</h3>' + df3.to_html(na_rep="",border=2,justify="center") + '<br>'
                            s9='<h3>3. Concentration in Rating Grades (Number Weighted)</h3>' + df4.to_html(index=False,border=2,justify="center") + '<br>'
                            s10='<h3>3. Concentration in Rating Grades (Exposure Weighted)</h3>' + df5.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            
                            res=''
                            if 'Q1' in list_of_keys:
                                res+=s1
                            if 'P1' in list_of_keys:
                                res+=s2
                            if 'P2' in list_of_keys:
                                res+=s3
                            if 'D1' in list_of_keys:
                                res+=s4
                            if 'S1' in list_of_keys:
                                res+=s5
                            if 'S2' in list_of_keys:
                                res+=s6
                            if 'S3' in list_of_keys:
                                res+=s7
                            if 'S4' in list_of_keys:
                                res+=s8
                            if 'S5' in list_of_keys:
                                res+=s9
                            if 'S6' in list_of_keys:
                                res+=s10
                
                                
                            _file= open("templates\Result1.html", 'a') 
                            _file.write('<center>' 
                                        +'<hr><hr><hr><h1>Model ('+M[i]+') Snapshot ('+S[j]+')</h1><hr><hr><hr>'
                                        +res
                                        +'</center>')
                            _file.close()
                            result+='<center>'+'<hr><hr><hr><h1>Model ('+M[i]+') Snapshot ('+S[j]+')</h1><hr><hr><hr>'+res+'</center>'
                        count+=1

                return result
            else:
                count=0
                result=''
                for i in range(len(M)-1):
                    for j in range(len(S_list)):
                        
                        sub_current = snaps_df[snaps_df['Model ID']==M[i]]#& (dfr2['Snapshot Date']==S_list[i])] 
                        sub_current = sub_current[sub_current['Snapshot Date']==S_list[j]]
                        
                        sub_prior = snaps_df[snaps_df['Model ID']==M[i]]#& (dfr2['Snapshot Date']==S_list[i])] 
                        sub_prior = sub_prior[sub_prior['Snapshot Date']==S_list_prior[j]]
                        
                        print("\nShape of the Snapshots Selected = ",sub_current.shape,sub_prior.shape,k,l)
            
                        
                        #calculate migration matrix in the form of list of lists (2d matrix), also calculates the possible grades in prior and current snapshot
                        migration_mat, grades_prior, grades_current = mig_mat_calculate(sub_prior,sub_current,"Final CRR") #Final CRR or Model CRR column name
                        
                        #modification in matrix into dataframe (to display)
                        migration_mat_df = mat_display(migration_mat,grades_prior,grades_current)
                        
                        #1st statbility metric calculation
                        df1=Customer_Migrations(migration_mat,migration_mat_df)
                        
                        #second stability metric calculation
                        df2, df3 = stability_of_migration_matrix(grades_prior,grades_current,migration_mat,migration_mat_df)
                        
                        #third stability metric calculation
                        CV_init=initial_data_df.loc[M[i],"Coefficient of variation"]
                        df4, df5,= concentration_in_RG(sub_current,migration_mat_df,CV_init)
                        
                        #Discriminatory Power (AUC) calculation
                        AUC_init=initial_data_df.loc[M[i],"AUC"]
                        df6 = discriminatory_pow(sub_prior,sub_current,AUC_init)
                        
                        #Predictive Ability calculation
                        df7, df8=pred_ability(sub_prior,sub_current,grades_prior)
                        
                        #Qualitative validation(Occ. of overrides) calculation
                        df9 = occ_of_override(sub_current)
                        
                        migration_mat_df.fillna('',inplace=True)
                        
                        if count==0:
                            s1='<h2>Qualitative Validation</h2><h3>Occurance of Overrides</h3>' + df9.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s2='<h2>Predictive Ability</h2><h3>PD back-testing (Portfolio level)</h3>' + df8.to_html(index=False,border=2,justify="center") + '<br>'
                            s3='<h2>Predictive Ability</h2><h3>PD back-testing (Grades level)</h3>' + df7.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s4='<h2>Discriminatory Power</h2><h3>Current AUC vs. AUC at initial validation/development</h3>' + df6.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s5='<h2>Stability</h2><h3>Migration Matrix</h3>' + migration_mat_df.to_html(border=2,justify="center") +'<br>'
                            s6='<h3>1. Customer Migrations</h3>' + df1.to_html(index=False,border=2,justify="center") + '<br>'
                            s7='<h3>2. Stability of Migration Matrix (Zij)</h3>' + df2.to_html(na_rep="",border=2,justify="center") + '<br>'
                            s8='<h3>2. Stability of Migration Matrix (PHIij)</h3>' + df3.to_html(na_rep="",border=2,justify="center") + '<br>'
                            s9='<h3>3. Concentration in Rating Grades (Number Weighted)</h3>' + df4.to_html(index=False,border=2,justify="center") + '<br>'
                            s10='<h3>3. Concentration in Rating Grades (Exposure Weighted)</h3>' + df5.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            
                            res=''
                            if 'Q1' in list_of_keys:
                                res+=s1
                            if 'P1' in list_of_keys:
                                res+=s2
                            if 'P2' in list_of_keys:
                                res+=s3
                            if 'D1' in list_of_keys:
                                res+=s4
                            if 'S1' in list_of_keys:
                                res+=s5
                            if 'S2' in list_of_keys:
                                res+=s6
                            if 'S3' in list_of_keys:
                                res+=s7
                            if 'S4' in list_of_keys:
                                res+=s8
                            if 'S5' in list_of_keys:
                                res+=s9
                            if 'S6' in list_of_keys:
                                res+=s10
                            if res=='':
                                ctypes.windll.user32.MessageBoxW(0, "\nPlease Select One of the metric.\nGo to First page(Files Upload) of the app or refresh the page.\nThank You...", "Warning",0x1000)
                            
                            
                            _file= open("templates\Result1.html", 'w')
                            _file.write('<center>' 
                                        +'<h1>Validation Results of IRB PD Model</h1><br>'
                                        +'<hr><hr><hr><hr><h1>Model ('+M[i]+') Snapshot ('+S_list[j]+')</h1><hr><hr><hr>'
                                        +res
                                        +'</center>')
                            _file.close()
                            result+='<center>'+'<h1>Validation Results of IRB PD Model</h1><br>'+'<hr><hr><hr><hr><h1>Model ('+M[i]+') Snapshot ('+S_list[j]+')</h1><hr><hr><hr>'+res+'</center>'
                        else:
                            s1='<h2>Qualitative Validation</h2><h3>Occurance of Overrides</h3>' + df9.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s2='<h2>Predictive Ability</h2><h3>PD back-testing (Portfolio level)</h3>' + df8.to_html(index=False,border=2,justify="center") + '<br>'
                            s3='<h2>Predictive Ability</h2><h3>PD back-testing (Grades level)</h3>' + df7.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s4='<h2>Discriminatory Power</h2><h3>Current AUC vs. AUC at initial validation/development</h3>' + df6.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            s5='<h2>Stability</h2><h3>Migration Matrix</h3>' + migration_mat_df.to_html(border=2,justify="center") +'<br>'
                            s6='<h3>1. Customer Migrations</h3>' + df1.to_html(index=False,border=2,justify="center") + '<br>'
                            s7='<h3>2. Stability of Migration Matrix (Zij)</h3>' + df2.to_html(na_rep="",border=2,justify="center") + '<br>'
                            s8='<h3>2. Stability of Migration Matrix (PHIij)</h3>' + df3.to_html(na_rep="",border=2,justify="center") + '<br>'
                            s9='<h3>3. Concentration in Rating Grades (Number Weighted)</h3>' + df4.to_html(index=False,border=2,justify="center") + '<br>'
                            s10='<h3>3. Concentration in Rating Grades (Exposure Weighted)</h3>' + df5.to_html(index=False,border=2,justify="center") + '<br><hr>'
                            
                            res=''
                            if 'Q1' in list_of_keys:
                                res+=s1
                            if 'P1' in list_of_keys:
                                res+=s2
                            if 'P2' in list_of_keys:
                                res+=s3
                            if 'D1' in list_of_keys:
                                res+=s4
                            if 'S1' in list_of_keys:
                                res+=s5
                            if 'S2' in list_of_keys:
                                res+=s6
                            if 'S3' in list_of_keys:
                                res+=s7
                            if 'S4' in list_of_keys:
                                res+=s8
                            if 'S5' in list_of_keys:
                                res+=s9
                            if 'S6' in list_of_keys:
                                res+=s10
                                
                            _file= open("templates\Result1.html", 'a') 
                            _file.write('<center>' 
                                        +'<hr><hr><hr><h1>Model ('+M[i]+') Snapshot ('+S_list[j]+')</h1><hr><hr><hr>'
                                        +res
                                        +'</center>')
                            _file.close()
                            result+='<center>'+'<hr><hr><hr><h1>Model ('+M[i]+') Snapshot ('+S_list[j]+')</h1><hr><hr><hr>'+res+'</center>'
                        count+=1
                
                return result
            
        elif 'All Snapshots' in S_list:
            count=0
            result=''
            for i in range(len(M_list)):
                for j in range(len(S)-1):
                    sub_current = snaps_df[snaps_df['Model ID']==M_list[i]]#& (dfr2['Snapshot Date']==S_list[i])] 
                    sub_current = sub_current[sub_current['Snapshot Date']==S[j]]
                    
                    sub_prior = snaps_df[snaps_df['Model ID']==M_list[i]]#& (dfr2['Snapshot Date']==S_list[i])] 
                    sub_prior = sub_prior[sub_prior['Snapshot Date']==S_prior[j]]
                    
                    print("\nShape of the Snapshots Selected = ",sub_current.shape,sub_prior.shape,k,l)
        
                    
                    #calculate migration matrix in the form of list of lists (2d matrix), also calculates the possible grades in prior and current snapshot
                    migration_mat, grades_prior, grades_current = mig_mat_calculate(sub_prior,sub_current,"Final CRR") #Final CRR or Model CRR column name
                    
                    #modification in matrix into dataframe (to display)
                    migration_mat_df = mat_display(migration_mat,grades_prior,grades_current)
                    
                    #1st statbility metric calculation
                    df1=Customer_Migrations(migration_mat,migration_mat_df)
                    
                    #second stability metric calculation
                    df2, df3 = stability_of_migration_matrix(grades_prior,grades_current,migration_mat,migration_mat_df)
                    
                    #third stability metric calculation
                    CV_init=initial_data_df.loc[M_list[i],"Coefficient of variation"]
                    df4, df5,= concentration_in_RG(sub_current,migration_mat_df,CV_init)
                    
                    #Discriminatory Power (AUC) calculation
                    AUC_init=initial_data_df.loc[M_list[i],"AUC"]
                    df6 = discriminatory_pow(sub_prior,sub_current,AUC_init)
                    
                    #Predictive Ability calculation
                    df7, df8=pred_ability(sub_prior,sub_current,grades_prior)
                    
                    #Qualitative validation(Occ. of overrides) calculation
                    df9 = occ_of_override(sub_current)
                    
                    migration_mat_df.fillna('',inplace=True)
                    
                    if count==0:
                        s1='<h2>Qualitative Validation</h2><h3>Occurance of Overrides</h3>' + df9.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s2='<h2>Predictive Ability</h2><h3>PD back-testing (Portfolio level)</h3>' + df8.to_html(index=False,border=2,justify="center") + '<br>'
                        s3='<h2>Predictive Ability</h2><h3>PD back-testing (Grades level)</h3>' + df7.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s4='<h2>Discriminatory Power</h2><h3>Current AUC vs. AUC at initial validation/development</h3>' + df6.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s5='<h2>Stability</h2><h3>Migration Matrix</h3>' + migration_mat_df.to_html(border=2,justify="center") +'<br>'
                        s6='<h3>1. Customer Migrations</h3>' + df1.to_html(index=False,border=2,justify="center") + '<br>'
                        s7='<h3>2. Stability of Migration Matrix (Zij)</h3>' + df2.to_html(na_rep="",border=2,justify="center") + '<br>'
                        s8='<h3>2. Stability of Migration Matrix (PHIij)</h3>' + df3.to_html(na_rep="",border=2,justify="center") + '<br>'
                        s9='<h3>3. Concentration in Rating Grades (Number Weighted)</h3>' + df4.to_html(index=False,border=2,justify="center") + '<br>'
                        s10='<h3>3. Concentration in Rating Grades (Exposure Weighted)</h3>' + df5.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        
                        res=''
                        if 'Q1' in list_of_keys:
                            res+=s1
                        if 'P1' in list_of_keys:
                            res+=s2
                        if 'P2' in list_of_keys:
                            res+=s3
                        if 'D1' in list_of_keys:
                            res+=s4
                        if 'S1' in list_of_keys:
                            res+=s5
                        if 'S2' in list_of_keys:
                            res+=s6
                        if 'S3' in list_of_keys:
                            res+=s7
                        if 'S4' in list_of_keys:
                            res+=s8
                        if 'S5' in list_of_keys:
                            res+=s9
                        if 'S6' in list_of_keys:
                            res+=s10
                        if res=='':
                            ctypes.windll.user32.MessageBoxW(0, "\nPlease Select One of the metric.\nGo to First page(Files Upload) of the app or refresh the page.\nThank You...", "Warning",0x1000)
                        
                        
                        _file= open("templates\Result1.html", 'w')
                        _file.write('<center>' 
                                    +'<h1>Validation Results of IRB PD Model</h1><br>'
                                    +'<hr><hr><hr><hr><h1>Model ('+M_list[i]+') Snapshot ('+S[j]+')</h1><hr><hr><hr>'
                                    +res
                                    +'</center>')
                        _file.close()
                        result+='<center>'+'<h1>Validation Results of IRB PD Model</h1><br>'+'<hr><hr><hr><hr><h1>Model ('+M_list[i]+') Snapshot ('+S[j]+')</h1><hr><hr><hr>'+res+'</center>'
                    else:
                        s1='<h2>Qualitative Validation</h2><h3>Occurance of Overrides</h3>' + df9.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s2='<h2>Predictive Ability</h2><h3>PD back-testing (Portfolio level)</h3>' + df8.to_html(index=False,border=2,justify="center") + '<br>'
                        s3='<h2>Predictive Ability</h2><h3>PD back-testing (Grades level)</h3>' + df7.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s4='<h2>Discriminatory Power</h2><h3>Current AUC vs. AUC at initial validation/development</h3>' + df6.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s5='<h2>Stability</h2><h3>Migration Matrix</h3>' + migration_mat_df.to_html(border=2,justify="center") +'<br>'
                        s6='<h3>1. Customer Migrations</h3>' + df1.to_html(index=False,border=2,justify="center") + '<br>'
                        s7='<h3>2. Stability of Migration Matrix (Zij)</h3>' + df2.to_html(na_rep="",border=2,justify="center") + '<br>'
                        s8='<h3>2. Stability of Migration Matrix (PHIij)</h3>' + df3.to_html(na_rep="",border=2,justify="center") + '<br>'
                        s9='<h3>3. Concentration in Rating Grades (Number Weighted)</h3>' + df4.to_html(index=False,border=2,justify="center") + '<br>'
                        s10='<h3>3. Concentration in Rating Grades (Exposure Weighted)</h3>' + df5.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        
                        res=''
                        if 'Q1' in list_of_keys:
                            res+=s1
                        if 'P1' in list_of_keys:
                            res+=s2
                        if 'P2' in list_of_keys:
                            res+=s3
                        if 'D1' in list_of_keys:
                            res+=s4
                        if 'S1' in list_of_keys:
                            res+=s5
                        if 'S2' in list_of_keys:
                            res+=s6
                        if 'S3' in list_of_keys:
                            res+=s7
                        if 'S4' in list_of_keys:
                            res+=s8
                        if 'S5' in list_of_keys:
                            res+=s9
                        if 'S6' in list_of_keys:
                            res+=s10
                            
                        _file= open("templates\Result1.html", 'a')
                        _file.write('<center>' 
                                    +'<hr><hr><hr><h1>Model ('+M_list[i]+') Snapshot ('+S[j]+')</h1><hr><hr><hr>'
                                    +res
                                    +'</center>')
                        _file.close()
                        result+='<center>'+'<hr><hr><hr><h1>Model ('+M_list[i]+') Snapshot ('+S[j]+')</h1><hr><hr><hr>'+res+'</center>'
                    count+=1
            
            return result
        else:
            count=0
            result=''
            for i in range(len(M_list)):
                for j in range(len(S_list)):
                    
                    sub_current = snaps_df[snaps_df['Model ID']==M_list[i]]#& (dfr2['Snapshot Date']==S_list[i])] 
                    sub_current = sub_current[sub_current['Snapshot Date']==S_list[j]]
                    
                    sub_prior = snaps_df[snaps_df['Model ID']==M_list[i]]#& (dfr2['Snapshot Date']==S_list[i])] 
                    sub_prior = sub_prior[sub_prior['Snapshot Date']==S_list_prior[j]]
                    
                    print("\nShape of the Snapshots Selected = ",sub_current.shape,sub_prior.shape,k,l)
        
                    
                    #calculate migration matrix in the form of list of lists (2d matrix), also calculates the possible grades in prior and current snapshot
                    migration_mat, grades_prior, grades_current = mig_mat_calculate(sub_prior,sub_current,"Final CRR") #Final CRR or Model CRR column name
                    
                    #modification in matrix into dataframe (to display)
                    migration_mat_df = mat_display(migration_mat,grades_prior,grades_current)
                    
                    #1st statbility metric calculation
                    df1=Customer_Migrations(migration_mat,migration_mat_df)
                    
                    #second stability metric calculation
                    df2, df3 = stability_of_migration_matrix(grades_prior,grades_current,migration_mat,migration_mat_df)
                    
                    #third stability metric calculation
                    CV_init=initial_data_df.loc[M_list[i],"Coefficient of variation"]
                    df4, df5,= concentration_in_RG(sub_current,migration_mat_df,CV_init)
                    
                    #Discriminatory Power (AUC) calculation
                    AUC_init=initial_data_df.loc[M_list[i],"AUC"]
                    df6 = discriminatory_pow(sub_prior,sub_current,AUC_init)
                    
                    #Predictive Ability calculation
                    df7, df8=pred_ability(sub_prior,sub_current,grades_prior)
                    
                    #Qualitative validation(Occ. of overrides) calculation
                    df9 = occ_of_override(sub_current)
                    
                    migration_mat_df.fillna('',inplace=True)
                    
                    if count==0:
                        s1='<h2>Qualitative Validation</h2><h3>Occurance of Overrides</h3>' + df9.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s2='<h2>Predictive Ability</h2><h3>PD back-testing (Portfolio level)</h3>' + df8.to_html(index=False,border=2,justify="center") + '<br>'
                        s3='<h2>Predictive Ability</h2><h3>PD back-testing (Grades level)</h3>' + df7.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s4='<h2>Discriminatory Power</h2><h3>Current AUC vs. AUC at initial validation/development</h3>' + df6.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s5='<h2>Stability</h2><h3>Migration Matrix</h3>' + migration_mat_df.to_html(border=2,justify="center") +'<br>'
                        s6='<h3>1. Customer Migrations</h3>' + df1.to_html(index=False,border=2,justify="center") + '<br>'
                        s7='<h3>2. Stability of Migration Matrix (Zij)</h3>' + df2.to_html(na_rep="",border=2,justify="center") + '<br>'
                        s8='<h3>2. Stability of Migration Matrix (PHIij)</h3>' + df3.to_html(na_rep="",border=2,justify="center") + '<br>'
                        s9='<h3>3. Concentration in Rating Grades (Number Weighted)</h3>' + df4.to_html(index=False,border=2,justify="center") + '<br>'
                        s10='<h3>3. Concentration in Rating Grades (Exposure Weighted)</h3>' + df5.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        
                        res=''
                        if 'Q1' in list_of_keys:
                            res+=s1
                        if 'P1' in list_of_keys:
                            res+=s2
                        if 'P2' in list_of_keys:
                            res+=s3
                        if 'D1' in list_of_keys:
                            res+=s4
                        if 'S1' in list_of_keys:
                            res+=s5
                        if 'S2' in list_of_keys:
                            res+=s6
                        if 'S3' in list_of_keys:
                            res+=s7
                        if 'S4' in list_of_keys:
                            res+=s8
                        if 'S5' in list_of_keys:
                            res+=s9
                        if 'S6' in list_of_keys:
                            res+=s10
                        if res=='':
                            ctypes.windll.user32.MessageBoxW(0, "\nPlease Select One of the metric.\nGo to First page(Files Upload) of the app or refresh the page.\nThank You...", "Warning",0x1000)
                        
                        
                        _file= open("templates\Result1.html", 'w') 
                        _file.write('<center>' 
                                    +'<h1>Validation Results of IRB PD Model</h1><br>'
                                    +'<hr><hr><hr><hr><h1>Model ('+M_list[i]+') Snapshot ('+S_list[j]+')</h1><hr><hr><hr>'
                                    +res
                                    +'</center>')
                        _file.close()
                        result+='<center>'+'<h1>Validation Results of IRB PD Model</h1><br>'+'<hr><hr><hr><hr><h1>Model ('+M_list[i]+') Snapshot ('+S_list[j]+')</h1><hr><hr><hr>'+res+'</center>'
                    else:
                        s1='<h2>Qualitative Validation</h2><h3>Occurance of Overrides</h3>' + df9.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s2='<h2>Predictive Ability</h2><h3>PD back-testing (Portfolio level)</h3>' + df8.to_html(index=False,border=2,justify="center") + '<br>'
                        s3='<h2>Predictive Ability</h2><h3>PD back-testing (Grades level)</h3>' + df7.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s4='<h2>Discriminatory Power</h2><h3>Current AUC vs. AUC at initial validation/development</h3>' + df6.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        s5='<h2>Stability</h2><h3>Migration Matrix</h3>' + migration_mat_df.to_html(border=2,justify="center") +'<br>'
                        s6='<h3>1. Customer Migrations</h3>' + df1.to_html(index=False,border=2,justify="center") + '<br>'
                        s7='<h3>2. Stability of Migration Matrix (Zij)</h3>' + df2.to_html(na_rep="",border=2,justify="center") + '<br>'
                        s8='<h3>2. Stability of Migration Matrix (PHIij)</h3>' + df3.to_html(na_rep="",border=2,justify="center") + '<br>'
                        s9='<h3>3. Concentration in Rating Grades (Number Weighted)</h3>' + df4.to_html(index=False,border=2,justify="center") + '<br>'
                        s10='<h3>3. Concentration in Rating Grades (Exposure Weighted)</h3>' + df5.to_html(index=False,border=2,justify="center") + '<br><hr>'
                        
                        res=''
                        if 'Q1' in list_of_keys:
                            res+=s1
                        if 'P1' in list_of_keys:
                            res+=s2
                        if 'P2' in list_of_keys:
                            res+=s3
                        if 'D1' in list_of_keys:
                            res+=s4
                        if 'S1' in list_of_keys:
                            res+=s5
                        if 'S2' in list_of_keys:
                            res+=s6
                        if 'S3' in list_of_keys:
                            res+=s7
                        if 'S4' in list_of_keys:
                            res+=s8
                        if 'S5' in list_of_keys:
                            res+=s9
                        if 'S6' in list_of_keys:
                            res+=s10
                            
                        _file= open("templates\Result1.html", 'a') 
                        _file.write('<center>' 
                                    +'<hr><hr><hr><h1>Model ('+M_list[i]+') Snapshot ('+S_list[j]+')</h1><hr><hr><hr>'
                                    +res
                                    +'</center>')
                        _file.close()
                        result+='<center>'+'<hr><hr><hr><h1>Model ('+M_list[i]+') Snapshot ('+S_list[j]+')</h1><hr><hr><hr>'+res+'</center>'
                    count+=1
            
            return result#render_template('Result1.html')#,tables=[migration_mat_df.to_html(), df1.to_html(),df2.to_html(),df3.to_html(),df4.to_html(),
                                                   #df5.to_html(),df6.to_html(),df7.to_html(),df8.to_html(),df9.to_html()],
                               #titles = ['na', 'DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'DF6', 'DF7', 'DF8','DF9'],result=result)




#Execution of the Application Starts from here (The Driver Function)                    
if __name__=="__main__":
    
    app.run(debug=False)