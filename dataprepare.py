#!/usr/bin/python
# Filename: offline-analysis-example.py
import os
import sys
import glob


"""
Offline analysis by replaying logs
"""

# Import MobileInsight modules
from mobile_insight.monitor import OfflineReplayer
from mobile_insight.analyzer import MsgLogger, LteRrcAnalyzer, WcdmaRrcAnalyzer, LteNasAnalyzer, UmtsNasAnalyzer, LtePhyAnalyzer, LteMacAnalyzer, LtePdcpAnalyzer, MmAnalyzer, LteMeasurementAnalyzer, LteRlcAnalyzer, MobilityMngt, RrcAnalyzer, ModemDebugAnalyzer


from xml.dom.minidom import parse

#import xml.etree.ElementTree as ET
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import xml.dom.minidom
import pandas as pd
import numpy as np

def ReadRawFile(filename):

    # Initialize a 3G/4G monitor
    src = OfflineReplayer()
    #src.set_input_path("./diag_log_20190701_032740_93cda9833356d6bc90e05d0f85b3efee_Google-PixelXL_311480.mi2log")
    src.set_input_path("./" + filename + ".mi2log")

    logger = MsgLogger()
    logger.set_decode_format(MsgLogger.XML)
    logger.set_dump_type(MsgLogger.FILE_ONLY)
    #logger.save_decoded_msg_as("./test.txt")
    logger.save_decoded_msg_as("./" + filename + ".txt")
    logger.set_source(src)

    
    # Analyzers

    lte_pdcp_analyzer = LtePdcpAnalyzer()
    lte_pdcp_analyzer.set_source(src)

    lte_rrc_analyzer = LteRrcAnalyzer()
    lte_rrc_analyzer.set_source(src)  # bind with the monitor

    wcdma_rrc_analyzer = WcdmaRrcAnalyzer()
    wcdma_rrc_analyzer.set_source(src)  # bind with the monitor

    lte_nas_analyzer = LteNasAnalyzer()
    lte_nas_analyzer.set_source(src)

    umts_nas_analyzer = UmtsNasAnalyzer()
    umts_nas_analyzer.set_source(src)

    lte_phy_analyzer = LtePhyAnalyzer()
    lte_phy_analyzer.set_source(src)

    lte_mac_analyzer = LteMacAnalyzer()
    lte_mac_analyzer.set_source(src)

    mm_analyzer = MmAnalyzer()
    mm_analyzer.set_source(src)

    lte_measurement_analyzer = LteMeasurementAnalyzer()
    lte_measurement_analyzer.set_source(src)

    lte_rlc_analyzer = LteRlcAnalyzer()
    lte_rlc_analyzer.set_source(src)

    #modem_Debug_Analyzer = ModemDebugAnalyzer()
    #modem_Debug_Analyzer.set_source(src)


    mobility_mngt = MobilityMngt()
    mobility_mngt.set_source(src)

    rrc_analyzer = RrcAnalyzer()
    rrc_analyzer.set_source(src)

    # Start the monitoring
    src.run()

def ReadXML(filename, output):
	f = open(filename + ".txt", "r")
	arr = []

	for line in f:
		if line[0:15] == '<dm_log_packet>' and line[-17:] == '</dm_log_packet>\n':
			#print '666666666'
			root = ET.fromstring(line)
			#print root.tag
			row = []
			for child in root:
				#print child.tag
				#print child.attrib
				if child.attrib == {'key': 'type_id'}:
					row.append(child.text)
				if child.attrib == {'key': 'timestamp'}:
					#print child.attrib
					row.append(child.text)
			arr.append(row)
		#convert to np array
	nparr = np.array(arr)

	dataset = pd.DataFrame({'type_id':nparr[:,0],'timestamp':nparr[:,1]})
	output = output.append(dataset)
	f.close()

	#print output.size
	return output

def DataToCSV(folders = None):

    dirpath = os.getcwd()
    
    if folders == None:
        #if no specific folders assigned
        subfolders = filter(os.path.isdir, os.listdir(dirpath))
    else:
        subfolders = folders

    #execute all subfolders
    for subfolder in subfolders:
        os.chdir(dirpath + "/" + subfolder)
        #print (os.getcwd())
        filenames = []
        for file in glob.glob("*.mi2log"):
            filenames.append(file[0:-7])

        filenames.sort()
        for files in filenames:
            ReadRawFile(files)
        newdf = pd.DataFrame()
        for files in filenames:
            newdf = ReadXML(files, newdf)

        newdf.sort_values(by=['timestamp'])
        foldername = os.path.basename(os.getcwd())
        newdf.to_csv(foldername + '.csv', sep='\t', encoding='utf-8')


if __name__== "__main__":

    DataToCSV()

    #llist = ['callin1_10s', 'callin2_10s', 'callin3_10s', 'callin4_10s', 'callin5_10s']
    #llist = ['sendsms2']
    #DataToCSV(llist) 
    

