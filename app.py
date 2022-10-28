
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objs as go

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import squarify
from datetime import datetime
import feature_engine
from feature_engine.outliers import Winsorizer

import warnings
warnings.filterwarnings("ignore")



df=pd.read_csv('OnlineRetail.csv', encoding= 'unicode_escape')


# Data Pre-processing
# drop na, null rows
df = df.dropna()
df = df[pd.notnull(df['CustomerID'])]
# drop negative rows
df = df[df.Quantity > 0]
df = df[df.UnitPrice > 0]

# convert data types
string_to_date = lambda x : datetime.strptime(x, "%d-%m-%Y %H:%M").date()

# Convert InvoiceDate from object to datetime format
df['InvoiceDate'] = df['InvoiceDate'].apply(string_to_date)
df['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')
df['CustomerID']=df['CustomerID'].astype('int64')

#RFM analysis
#Create RFM analysis for each customers
#Convert string to date, get max date of dataframe

max_date=df['InvoiceDate'].max().date()
Recency=lambda x: (max_date-x.max().date()).days
Frequency=lambda x: len(x.unique())
Monetary=lambda x: round((x*df.Quantity).sum(), 2)
df_RFM=df.groupby('CustomerID').agg({'InvoiceDate': Recency,'InvoiceNo': Frequency,'UnitPrice': Monetary})

#Rename the columns of dataframe
df_RFM.columns=['Recency', 'Frequency', 'Monetary']
df_RFM=df_RFM.sort_values('Monetary', ascending=False)


# take logarithm of df_RFM data to remove skewness
df_RFM_log=np.log(df_RFM[['Recency','Frequency', 'Monetary']]+1)


#STEP-2
df_RFM_scaled=df_RFM_log.copy()

#scale the data
scaler=StandardScaler()

# transform into the dataframe
columns_name=['RecencyScale','FrequencyScale', 'MonetaryScale']
df_RFM_scaled=scaler.fit_transform(df_RFM_scaled)
df_RFM_scaled=pd.DataFrame(df_RFM_scaled, index=df_RFM.index, columns=columns_name)

## Calculate RFM quartiles
r_labels=range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gán gán nhỏ, ngược lại thì gán nhản lớn
f_labels=range(1,5)
m_labels=range(1,5)

# Assign these labels to 4 equal percentile groups

r_groups=pd.qcut(df_RFM_scaled['RecencyScale'].rank(method='first'), q=4, labels=r_labels)
f_groups=pd.qcut(df_RFM_scaled['FrequencyScale'].rank(method='first'), q=4, labels=f_labels) 
m_groups=pd.qcut(df_RFM_scaled['MonetaryScale'].rank(method='first'), q=4, labels=m_labels)

# Create new columns R, F, M
df_RFM_scaled=df_RFM_scaled.assign(R=r_groups.values, F=f_groups.values, M=m_groups.values)


# Concat RFM quartile values to create RFM Segments
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
df_RFM_scaled['RFM_Segment'] = df_RFM_scaled.apply(join_rfm, axis=1)


##4.5 Calculate RFM score and level
# Calculate RFM score
df_RFM_scaled['RFM_Score']=df_RFM_scaled[['R', 'F', 'M']].sum(axis=1)

# assign labels from total score
score_labels = ['Green', 'Bronze', 'Silver', 'Gold']
score_groups = pd.qcut(df_RFM_scaled.RFM_Score, q = 4, labels = score_labels)
df_RFM_scaled['RFM_Level'] = score_groups.values


# Number of segments

df_RFM_scaled['RFM_Level'].value_counts()
df_RFM_merge=df_RFM_scaled.merge(df_RFM, left_index=True, right_index=True)



# Calculate mean values for each segment
# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_agg = df_RFM_merge.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count', 'sum']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count', 'MonetarySum']
rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

# Reset the index
rfm_agg = rfm_agg.reset_index()


# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_scale_agg = df_RFM_merge.groupby('RFM_Level').agg({
    'RecencyScale': 'mean',
    'FrequencyScale': 'mean',
    'MonetaryScale': ['mean', 'count', 'sum']}).round(0)

rfm_scale_agg.columns = rfm_scale_agg.columns.droplevel()
rfm_scale_agg.columns = ['RecencyScaleMean','FrequencyScaleMean','MonetaryScaleMean', 'Count', 'MonetaryScaleSum']
rfm_scale_agg['Percent'] = round((rfm_scale_agg['Count']/rfm_scale_agg.Count.sum())*100, 2)

# Reset the index
rfm_scale_agg = rfm_scale_agg.reset_index()


rfm_agg_merge=rfm_agg.merge(rfm_scale_agg[['RecencyScaleMean','FrequencyScaleMean','MonetaryScaleMean', 'MonetaryScaleSum']], left_index=True, right_index=True)

order=['RFM_Level', 'RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count', 'Percent', 
       'RecencyScaleMean', 'FrequencyScaleMean', 'MonetaryScaleMean', 'MonetaryScaleSum','MonetarySum']

rfm_agg_merge = rfm_agg_merge.reindex(columns=order)



# TreeMap
# #Create our plot and resize it.
# fig1 = plt.gcf()
# ax = fig1.add_subplot()
# fig1.set_size_inches(25, 10)

# colors_dict = {'ACTIVE':'yellow','BIG SPENDER':'royalblue', 'LIGHT':'cyan',
#                'LOST':'red', 'LOYAL':'purple', 'POTENTIAL':'green', 'STARS':'gold'}

# squarify.plot(sizes=rfm_agg_merge['Count'],
#               text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
#               color=colors_dict.values(),
#               label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg_merge.iloc[i])
#                       for i in range(0, len(rfm_agg_merge))], alpha=0.5 )


# plt.title("Customers Segments",fontsize=26,fontweight="bold", x=0.5, y=1.05)
# plt.axis('off')
# st.pyplot(fig1)

# """Scatter Plot (RFM)"""
# import plotly.express as px

# fig2=px.scatter(rfm_agg_merge, x='RecencyMean', y='MonetaryMean', size='FrequencyMean', color='RFM_Level',
#                hover_name='RFM_Level', size_max=100)
# st.plotly_chart(fig2)

# """3d Scatter Plot (RFM)"""
# import plotly.express as px

# fig3 = px.scatter_3d(df_RFM_merge, x='Recency', y='Frequency', z='Monetary',
#                     color = 'RFM_Level', opacity=0.5,
#                     color_discrete_map = colors_dict)
# fig3.update_traces(marker=dict(size=5),
                  
#                   selector=dict(mode='markers'))
# st.plotly_chart(fig3)

# STEP-3
# Kmeans clusters with the Elbow Method
# GMM, hiarical, Kmean
df_now = df_RFM_merge[['RecencyScale','FrequencyScale','MonetaryScale']]


from sklearn.cluster import KMeans
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_now)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

# fig4 = plt.figure(figsize=(10, 4))
# plt.title('The Elbow Method')
# plt.xlabel('k')
# plt.ylabel('SSE')
# sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
# st.pyplot(fig4)

# Build model with k=4
model = KMeans(n_clusters=4, random_state=42)
model.fit(df_now)
# model.labels_.shape

df_now["Cluster"] = model.labels_

df_now.groupby('Cluster').agg({
    'RecencyScale':'mean',
    'FrequencyScale':'mean',
    'MonetaryScale':['mean', 'count']}).round(2)



# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_agg2 = df_now.groupby('Cluster').agg({
    'RecencyScale': 'mean',
    'FrequencyScale': 'mean',
    'MonetaryScale': ['mean', 'count', 'sum']}).round(0)

rfm_agg2.columns = rfm_agg2.columns.droplevel()
rfm_agg2.columns = ['RecencyScaleMean','FrequencyScaleMean','MonetaryScaleMean', 'Count', 'MonetaryScaleSum']
rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

# Reset the index
rfm_agg2 = rfm_agg2.reset_index()

# Change thr Cluster Columns Datatype into discrete values
rfm_agg2['Cluster'] = rfm_agg2['Cluster'].astype('str')


# # #scale the data
sc_recency=StandardScaler()
recency_scale= df_RFM_log['Recency'].values #convert to numpy array
recency_scaled = sc_recency.fit_transform(recency_scale.reshape(-1,1))
recency_inversed = sc_recency.inverse_transform(recency_scaled)
recencymean_scale=rfm_agg2['RecencyScaleMean'].values.reshape(-1,1)
recencymean_inversed = sc_recency.inverse_transform(recencymean_scale)
recencymean=np.exp(recencymean_inversed)-1


# # #scale the data
sc_frequency=StandardScaler()
frequency_scale= df_RFM_log['Frequency'].values #convert to numpy array
frequency_scaled = sc_frequency.fit_transform(frequency_scale.reshape(-1,1))
frequency_inversed = sc_frequency.inverse_transform(frequency_scaled)
frequencymean_scale=rfm_agg2['FrequencyScaleMean'].values.reshape(-1,1)
frequencymean_inversed = sc_frequency.inverse_transform(frequencymean_scale)
frequencymean=np.exp(frequencymean_inversed)-1


# # #scale the data
sc_monetary=StandardScaler()
monetary_scale= df_RFM_log['Monetary'].values #convert to numpy array
monetary_scaled = sc_monetary.fit_transform(monetary_scale.reshape(-1,1))
monetary_inversed = sc_monetary.inverse_transform(monetary_scaled)
monetarymean_scale=rfm_agg2['MonetaryScaleMean'].values.reshape(-1,1)
monetarymean_inversed = sc_monetary.inverse_transform(monetarymean_scale)
monetarymean=np.exp(monetarymean_inversed)-1


monetarysum_scale=rfm_agg2['MonetaryScaleSum'].values.reshape(-1,1)
monetarysum_inversed = sc_monetary.inverse_transform(monetarysum_scale)
monetarysum=np.exp(monetarysum_inversed)-1


r=pd.DataFrame(recencymean, columns=['RecencyMean'])
f=pd.DataFrame(frequencymean, columns=['FrequencyMean'])
m=pd.DataFrame(monetarymean, columns=['MonetaryMean'])
s=pd.DataFrame(monetarymean, columns=['MonetarySum'])

rfm_agg3=pd.concat([rfm_agg2,r,f,m,s], axis=1)

order=['Cluster', 'RecencyMean', 'FrequencyMean','MonetaryMean', 'Count', 'Percent', 'RecencyScaleMean', 'FrequencyScaleMean', 'MonetaryScaleMean', 'MonetaryScaleSum', 'MonetarySum']

rfm_agg3 = rfm_agg3.reindex(columns=order)




# Customer segmentation
#Create our plot and resize it.
# fig5 = plt.gcf()
# ax = fig5.add_subplot()
# fig5.set_size_inches(14, 10)
# list=[0, 6, 7, 8, 4, 5]

# colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
#                'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

# squarify.plot(sizes=rfm_agg3['Count'],
#               text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
#               color=colors_dict2.values(),
#               label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg3.iloc[i]) for i in range(0, len(rfm_agg3))], alpha=0.5 )


# plt.title("Customers Segments",fontsize=26,fontweight="bold")
# plt.axis('off')
# st.pyplot(fig5)


# import plotly.express as px
# fig = px.scatter_3d(rfm_agg3, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
#                     color = 'Cluster', opacity=0.3)
# fig.update_traces(marker=dict(size=20),
                 
#                   selector=dict(mode='markers'))
# fig.show()


# import plotly.express as px
# fig = px.scatter(rfm_agg3, x="RecencyMean", y="MonetaryMean", size='Count', color="Cluster",
#            hover_name="Cluster", size_max=100)
# fig.show()


# import plotly.express as px
# fig=px.scatter(rfm_agg_merge, x='RecencyMean', y='MonetaryMean', size='FrequencyMean', color='RFM_Level',
#                hover_name='RFM_Level', size_max=100)
# fig.show()

df_RFM_merge["Cluster"] = model.labels_

# Visualization
##5.1 Snake Plot
# assign cluster column
rfm_data=df_RFM_merge[['RecencyScale', 'FrequencyScale', 'MonetaryScale', 'Cluster','RFM_Level']]
rfm_data.reset_index(inplace=True)


# melt the dataframe
rfm_melted=pd.melt(rfm_data, id_vars=['CustomerID', 'RFM_Level', 'Cluster'], var_name='Metrics', value_name='Value')


# # Create snake plot with RFM
# fig6 = plt.figure(figsize=(10, 4))
# sns.lineplot(x='Metrics', y='Value', hue='RFM_Level', data=rfm_melted)
# plt.title('Snake Plot of RFM')
# plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.05))
# st.pyplot(fig6)

# # Create snake plot with Kmeans
# fig7 = plt.figure(figsize=(10, 4))
# sns.lineplot(x='Metrics', y='Value', hue='Cluster', data=rfm_melted)
# plt.title('Snake Plot of Cluster')
# plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
# st.pyplot(fig7)


# st.markdown("""RFM_Level and Cluster by Kmean shows similar trend.
# - cluster 1: match with Gold
# - cluster 0: match with Silver
# - cluster 2: match with Bronze
# - cluster 3: match with Green
# """)

##5.2 Heatmap
#Heatmap is efficient for comparing the standardized values.


# the mean value for each RFM_Level
rfm_avg = df_RFM_merge.groupby('RFM_Level').mean()
rfm_avg =rfm_avg.drop(['RecencyScale',	'FrequencyScale',	'MonetaryScale',	'RFM_Score', 'Cluster'], axis=1)

# the mean value in total 
total_rfm_avg = df_RFM_merge.iloc[:, 9:12].mean()


# the proportional mean value
prop_rfm = rfm_avg/total_rfm_avg - 1


# heatmap
# sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True)
# plt.title('Heatmap of RFM quantile')
# plt.plot()

# Gold show high frequency purchase product, big buy and recently"""

# the mean value for each cluster
cluster_avg = df_RFM_merge.groupby('Cluster').mean()
cluster_avg =cluster_avg.drop(['RecencyScale',	'FrequencyScale',	'MonetaryScale',	'RFM_Score'], axis=1)


# the proportional mean value
prop_cluster = cluster_avg/total_rfm_avg - 1


# heatmap
# sns.heatmap(prop_cluster, cmap= 'Blues', fmt= '.2f', annot = True)
# plt.title('Heatmap of K-Means')
# plt.plot()

# df_RFM_merge.Cluster

# Cluster 1 show high frequency perchase product, big buy and recently ~ STARS

#6.Conclusion

# This exercise has been addressed the customer segmentation by using RFM and Kmean. as the result, three clusters were grouped namely 'Cluster 0', 'Cluster 1', 'Cluster 2'.
##'CLuster 1' is the most best customers, made biggest purchases so we can not lose them but we need to reward them for long term commitment

## 'Cluster 0' is the great customers who purchase regularly and frequently. They may become out new STARS

## 'Cluster 2' is the poorest, newest customers not spend much for purchasing our product. New program can be launched out to extract these group.


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Wellcome to our page',
                          
                          ['Introduction',
                           'RFM Calculation',
                           'Build Kmean model',
                           'Evaluation'],
                          icons=['book', 'calculator','person', 'sun'],
                          default_index=0)
    
    


    # Information about us
    st.sidebar.title("About us")
    st.sidebar.info(
        """
        This web [app](....) is maintained by [Quách Thu Vũ & Thái Văn Đức]. 
        Học Viên lớp LDS0_K279 | THTH DHKHTN |
    """
    )




if (selected == 'Introduction'):
    st.title("Customer Segmetation using RFM and Kmean clustering")
    st.header("Step 1. Introduction/Bussiness Understanding")

    st.subheader("“Customer Segmentation” là gì?")
    st.write("""
    Phân khúc/nhóm/cụm khách hàng (market segmentation còn được gọi là phân khúc thị trường) là quá trình 
    nhóm các khách hàng lại với nhau dựa trên các đặc điểm chung. Nó phân chia và nhóm khách hàng thành 
    các nhóm nhỏ theo đặc điểm địa lý, nhân khẩu học, tâm lý học, hành vi (geographic, demographic, .
    psychographic, behavioral) và các đặc điểm khác.

    Các nhà tiếp thị sử dụng kỹ thuật này để nhắm mục tiêu khách hàng thông qua việc cá nhân hóa, khi họ muốn 
    tung ra các chiến dịch quảng cáo, truyền thông, thiết kế một ưu đãi hoặc khuyến mãi mới, và cũng để bán hàng
    """)
    # Images
    from PIL import Image 
    img = Image.open("RFM_1.jpg")
    st.image(img,width=700,caption='Streamlit Images')

    st.header("Steps to prepare Customer Segmetation using RFM and KMean model")
    st.markdown("""
    - Trực quan hóa dữ liệu
    - Lựa chọn thuật toán cho bài toán phân cụm
    - Xây dựng model
    - Đánh giá model 
    - Báo cáo kết quả
    """)

    st.header('The Online Retail dataset')
    st.info("""This is a transnational data set which contains all the transactions occurring 
    between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail""")
    st.markdown("""
    **Attribute Information:**
    - InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
    - StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
    - Description: Product (item) name. Nominal.
    - Quantity: The quantities of each product (item) per transaction. Numeric.
    - InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
    - UnitPrice: Unit price. Numeric, Product price per unit in sterling.
    - CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
    - Country: Country name. Nominal, the name of the country where each customer resides.
    """)
    st.success('Online Retail dataset has information shown as dataframe below')
    st.write(df.shape)
    st.dataframe(df.head())

elif (selected == 'RFM Calculation'):
    st.header('Step 2. Calculate RFM score')
    st.dataframe(df_RFM_merge)

    st.header('Four customer segments based on RFM Level')
    st.dataframe(rfm_agg_merge)

    #TreeMap
    fig1 = plt.gcf()
    ax = fig1.add_subplot()
    fig1.set_size_inches(25, 10)

    colors_dict = {'ACTIVE':'yellow','BIG SPENDER':'royalblue', 'LIGHT':'cyan',
                'LOST':'red', 'LOYAL':'purple', 'POTENTIAL':'green', 'STARS':'gold'}

    squarify.plot(sizes=rfm_agg_merge['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg_merge.iloc[i])
                        for i in range(0, len(rfm_agg_merge))], alpha=0.5 )

    plt.title("Customers Segments",fontsize=26,fontweight="bold", x=0.5, y=1.05)
    plt.axis('off')
    st.pyplot(fig1)

    #Scatter Plot (RFM)
    import plotly.express as px
    fig2=px.scatter(rfm_agg_merge, x='RecencyMean', y='MonetaryMean', size='FrequencyMean', color='RFM_Level',
                hover_name='RFM_Level', size_max=100)
    st.plotly_chart(fig2)

    #3d Scatter Plot (RFM)
    import plotly.express as px
    fig3 = px.scatter_3d(df_RFM_merge, x='Recency', y='Frequency', z='Monetary',
                        color = 'RFM_Level', opacity=0.5,
                        color_discrete_map = colors_dict)
    fig3.update_traces(marker=dict(size=5),
                    
                    selector=dict(mode='markers'))
    st.plotly_chart(fig3)

elif (selected == 'Build Kmean model'):
    st.header('Step 3. Build Kmean classification model')
    st.write('This session describe the process to build Kmean classification model to segment customer into appropriate group of customber to be taken care of')

    # elbow plot
    st.write('Determine the number of customer class by elbow curve')
    fig4 = plt.figure(figsize=(10, 4))
    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    st.pyplot(fig4)

    st.header('Four customer segments based on RFM Level and KMean')
    st.dataframe(rfm_agg3)

    st.write('TreeMap show the 4 class of customber and their characteristics')
    fig5 = plt.figure(figsize=(10, 4))
    # ax = fig5.add_subplot()
    fig5.set_size_inches(14, 10)
    list=[0, 6, 7, 8, 4, 5]

    colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

    squarify.plot(sizes=rfm_agg3['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict2.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg3.iloc[i]) for i in range(0, len(rfm_agg3))], alpha=0.5 )


    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    plt.axis('off')
    st.pyplot(fig5)


else: 
    st.header('Step 4. Evaluation the customer segmentation')
    st.write('The snake plot below show the pattern of 4 customer segmentation defined by RFM score')
    # Create snake plot with RFM
    fig6 = plt.figure(figsize=(15, 7))
    sns.lineplot(x='Metrics', y='Value', hue='RFM_Level', data=rfm_melted)
    plt.title('Snake Plot of RFM')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
    st.pyplot(fig6)

    # Create snake plot with Kmeans
    st.write('The snake plot below show the pattern of 4 customer segmentation defined by KMean clustering')
    fig7 = plt.figure(figsize=(10, 4))
    sns.lineplot(x='Metrics', y='Value', hue='Cluster', data=rfm_melted)
    plt.title('Snake Plot of Cluster')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
    st.pyplot(fig7)


    st.write("RFM_Level and Cluster by Kmean shows similar trend.")
    st.markdown('- cluster 1: match with Gold')
    st.markdown('- cluster 0: match with Silver')
    st.markdown('- cluster 2: match with Bronze')
    st.markdown('- cluster 3: match with Green')