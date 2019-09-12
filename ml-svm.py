# -*- coding: utf-8 -*
#!/usr/bin/python



import os
import sys
import pandas as pd 
import numpy as np
from sklearn.svm import SVC
from itertools import chain
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt

support_messages = {'LTE_MAC_Configuration':1, 'LTE_MAC_DL_Transport_Block':2, 'LTE_MAC_Rach_Attempt':3, 'LTE_MAC_Rach_Trigger':4, 'LTE_MAC_UL_Buffer_Status_Internal':5, 'LTE_MAC_UL_Transport_Block':7, 'LTE_MAC_UL_Tx_Statistics':8, 'LTE_NAS_EMM_OTA_Incoming_Packet':11, 'LTE_NAS_EMM_OTA_Outgoing_Packet':12, 'LTE_NAS_EMM_State':13, 'LTE_NAS_ESM_OTA_Incoming_Packet':14, 'LTE_NAS_ESM_OTA_Outgoing_Packet':15, 'LTE_NAS_ESM_State':16, 'LTE_PDCP_DL_Cipher_Data_PDU':17, 'LTE_PDCP_DL_Config':18, 'LTE_PDCP_DL_Ctrl_PDU':19, 'LTE_PDCP_DL_SRB_Integrity_Data_PDU':20, 'LTE_PDCP_DL_Stats':21, 'LTE_PDCP_UL_Cipher_Data_PDU':22, 'LTE_PDCP_UL_Config':23, 'LTE_PDCP_UL_Ctrl_PDU':24, 'LTE_PDCP_UL_Data_PDU':25, 'LTE_PDCP_UL_SRB_Integrity_Data_PDU':26, 'LTE_PDCP_UL_Stats':27, 'LTE_PHY_BPLMN_Cell_Confirm':28, 'LTE_PHY_BPLMN_Cell_Request':29, 'LTE_PHY_Connected_Mode_Intra_Freq_Meas':30, 'LTE_PHY_Connected_Mode_Neighbor_Measurement':31, 'LTE_PHY_Idle_Neighbor_Cell_Meas':32, 'LTE_PHY_Inter_RAT_CDMA_Measurement':33, 'LTE_PHY_Inter_RAT_Measurement':34, 'LTE_PHY_PDCCH_Decoding_Result':35, 'LTE_PHY_PDCCH_PHICH_Indication_Report':36, 'LTE_PHY_PDSCH_Decoding_Result':37, 'LTE_PHY_PDSCH_Packet':38, 'LTE_PHY_PDSCH_Stat_Indication':39, 'LTE_PHY_PUCCH_CSF':40, 'LTE_PHY_PUCCH_Power_Control':41, 'LTE_PHY_PUCCH_Tx_Report':42, 'LTE_PHY_PUSCH_CSF':43, 'LTE_PHY_PUSCH_Power_Control':44, 'LTE_PHY_PUSCH_Tx_Report':45, 'LTE_PHY_RLM_Report':46, 'LTE_PHY_Serv_Cell_Measurement':47, 'LTE_PHY_Serving_Cell_COM_Loop':48, 'LTE_PHY_System_Scan_Results':49, 'LTE_RLC_DL_AM_All_PDU':50, 'LTE_RLC_DL_Config_Log_Packet':51, 'LTE_RLC_DL_Stats':52, 'LTE_RLC_UL_AM_All_PDU':53, 'LTE_RLC_UL_Config_Log_Packet':54, 'LTE_RLC_UL_Stats':55, 'LTE_RRC_CDRX_Events_Info':56, 'LTE_RRC_MIB_Message_Log_Packet':57, 'LTE_RRC_MIB_Packet':58, 'LTE_RRC_OTA_Packet':59, 'LTE_RRC_Serv_Cell_Info':60, 'Srch_TNG_1x_Searcher_Dump':61, 'CDMA_Paging_Channel_Message':62}


support_messages_PHY = {'LTE_PHY_BPLMN_Cell_Confirm':1, 'LTE_PHY_BPLMN_Cell_Request':2, 'LTE_PHY_Connected_Mode_Intra_Freq_Meas':3, 'LTE_PHY_Connected_Mode_Neighbor_Measurement':4, 'LTE_PHY_Idle_Neighbor_Cell_Meas':5, 'LTE_PHY_Inter_RAT_CDMA_Measurement':6, 'LTE_PHY_Inter_RAT_Measurement':7, 'LTE_PHY_PDCCH_Decoding_Result':8, 'LTE_PHY_PDCCH_PHICH_Indication_Report':9, 'LTE_PHY_PDSCH_Decoding_Result':10, 'LTE_PHY_PDSCH_Packet':11, 'LTE_PHY_PDSCH_Stat_Indication':12, 'LTE_PHY_PUCCH_CSF':13, 'LTE_PHY_PUCCH_Power_Control':14, 'LTE_PHY_PUCCH_Tx_Report':15, 'LTE_PHY_PUSCH_CSF':16, 'LTE_PHY_PUSCH_Power_Control':17, 'LTE_PHY_PUSCH_Tx_Report':18, 'LTE_PHY_RLM_Report':19, 'LTE_PHY_Serv_Cell_Measurement':20, 'LTE_PHY_Serving_Cell_COM_Loop':21, 'LTE_PHY_System_Scan_Results':22 }




def convert(llist):
    temp = []
    for item in llist:
        if item != 'Modem_debug_message':
             temp.append(support_messages[item])

    return temp


def loaddata(filepaths):
    allrows = []
    for filepath in filepaths:
        X = []
        data = pd.read_csv(filepath, sep='\t', engine='python')
        #select the col 'type_id'
        col = data['type_id']
        for itm in col: 
            #if itm != 'Modem_debug_message':
            if itm in support_messages_PHY:
                X.append(support_messages_PHY[itm])
        allrows.append(X)
    return allrows

if __name__ == "__main__":
    
    

    #data = pd.read_csv("./sendsms3/sendsms3.csv", sep='\t', engine='python')
    #col = data['type_id']

    filepaths = ["./sendsms2/sendsms2.csv", "./sendsms3/sendsms3.csv", "./sendsms4/sendsms4.csv", "./sendsms5/sendsms5.csv", "./receivesms2/receivesms2.csv", "./receivesms3/receivesms3.csv", "./receivesms4/receivesms4.csv", "./receivesms5/receivesms5.csv", "./receivesms6/receivesms6.csv", "./callout1_10s/callout1_10s.csv", "./callout2_10s/callout2_10s.csv", "./callout3_10s/callout3_10s.csv", "./callout4_10s/callout4_10s.csv", "./callout5_10s/callout5_10s.csv", "./receivephonecall_15s/receivephonecall_15s.csv"]
    #, "./callin2_10s/callin2_10s.csv", "./callin3_10s/callin3_10s.csv", "./callin4_10s/callin4_10s.csv"]
    allrows = loaddata(filepaths)
    
    df = pd.DataFrame(allrows)


    #fill n/a by -1
    df.fillna(-1, inplace=True)

    #convert dataframe to np.array
    dfnp = df.to_numpy()


    y = np.array([1,1,1,1,1,1,1,1,1,2,2,2,2,2,2])
    #clf = svm.SVC(kernel='linear', C=1)
    #clf = SVC(C=1, kernel='rbf', gamma=1)
    #clf = SVC(gamma='auto')
    #clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    #clf = tree.DecisionTreeClassifier()


    #####################################################
    #set lgbm parameters
    """
    params = {
            'boosting_type': 'gbdt',
            'boosting': 'dart',
            'objective': 'binary',
            #'metric': 'binary_logloss',
            'metric': ('auc', 'logloss'),
 
            'learning_rate': 0.01,
            'num_leaves':1,
            'max_depth':3,
 
            'max_bin':10,
            'min_data_in_leaf':8,
 
            'feature_fraction': 0.6,
            'bagging_fraction': 1,
            'bagging_freq':0,
 
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_split_gain': 0
    }
    """

    

    evals_result = {}

    lgb_train = lgb.Dataset(dfnp, y, free_raw_data=False)
    lgb_eval = lgb.Dataset(dfnp, y, reference=lgb_train,free_raw_data=False)


    gbm = lgb.train(params,                    
                lgb_train,                  
                #num_boost_round=2000,       
                num_boost_round=20,
                valid_sets=lgb_eval,        
                evals_result=evals_result,
                #early_stopping_rounds=30
                )   # 早停系数

    preds_offline = gbm.predict(dfnp, num_iteration=gbm.best_iteration) 
    print preds_offline
    

    print evals_result
    
    import matplotlib.pyplot as plt

    lgb.create_tree_digraph(gbm, tree_index=1)

    """
    
    ax = lgb.plot_metric(evals_result, metric='auc')#metric的值与之前的params里面的值对应
    plt.show()
    
    
    ax = lgb.plot_tree(gbm, tree_index=0, figsize=(20, 8), show_info=['split_gain'])
    plt.show()
    """
    ########################################################


    """
    scores = cross_val_score(clf, dfnp, y, cv=2)
    print scores                        
    
    #test = np.array([[1, 2, 3], [4, 5, 6]])
    #y = np.array([1, 2])
    clf.fit(dfnp, y) 
    print clf.score(dfnp, y)


    #load test data
    test_filepaths = ["./sendsms3/sendsms3.csv", "./sendsms4/sendsms4.csv", "./sendsms5/sendsms5.csv", "./receivesms2/receivesms2.csv", "./receivesms3/receivesms3.csv", "./receivesms4/receivesms4.csv", "./receivesms5/receivesms5.csv", "./receivesms6/receivesms6.csv", "./callout1_10s/callout1_10s.csv", "./callout2_10s/callout2_10s.csv", "./callout3_10s/callout3_10s.csv", "./callout4_10s/callout4_10s.csv", "./callout5_10s/callout5_10s.csv", "./receivephonecall_15s/receivephonecall_15s.csv"]
    #, "./callin1_10s/callin1_10s.csv", "./callin2_10s/callin2_10s.csv", "./callin3_10s/callin3_10s.csv", "./callin4_10s/callin4_10s.csv" , "./callin5_10s/callin5_10s.csv"]
    df_test = pd.DataFrame(loaddata(test_filepaths))
    df_test.fillna(-1, inplace=True)
    dfnp_test = df_test.to_numpy()
    #print dfnp_test[-1]

    print(clf.predict(dfnp_test))
    


    df_test.to_csv('dfnp_test.csv', sep='\t', encoding='utf-8')
    
    """

    """
    with open("iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    """