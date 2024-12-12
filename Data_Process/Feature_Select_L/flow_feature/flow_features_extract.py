import collections
import os
import re
from socket import AF_INET
import dpkt
import numpy as np
import pandas as pd
from _socket import inet_ntop

class Flow(object):

    def __init__(self):
        """
            param
            -----------
            main: str
              'payload' means the main lengths sequence and timestampes sequence refer to packets with non-zero payload, the sequences will fitler out zero payload packets.
              'ip'   means the main lengths sequence and timestamps sequence refer to any packets, it will not filter any packets.
        """
        self.src = None
        self.sport = None
        self.dst = None
        self.dport = None
        self.protocol = None
        self.pkt_lengths = list()
        self.timestamps = list()

    # Add new packet to flow #
    def add(self, ts, buf):

        try:
            eth = dpkt.ethernet.Ethernet(buf)
            if not (isinstance(eth.data, dpkt.ip.IP)):
                return
            ip = eth.data

            transf = ip.data

            if (bytes(ip)[9] == 0x06):
                self.protocol = 6
            elif(bytes(ip)[9] == 0x11):
                self.protocol = 17
            else:
                self.protocol = 0

            port_a = transf.sport
            port_b = transf.dport

            if (clientIp_judge(inet_ntop(AF_INET, ip.src)) and clientIp_judge(inet_ntop(AF_INET, ip.dst)) == False):
                self.src, self.dst = ip.src, ip.dst
                self.sport, self.dport = port_a, port_b
            elif (clientIp_judge(inet_ntop(AF_INET, ip.dst)) and clientIp_judge(inet_ntop(AF_INET, ip.src)) == False):
                self.src, self.dst = ip.dst, ip.src
                self.sport, self.dport = port_b, port_a
            else:
                if port_a > port_b:
                    self.src, self.dst = ip.src, ip.dst
                    self.sport, self.dport = port_a, port_b
                else:
                    self.src, self.dst = ip.dst, ip.src
                    self.sport, self.dport = port_b, port_a

            if not len(transf.data):
                return

            self.timestamps.append(ts)
            self.pkt_lengths.append(len(buf) if (ip.src, transf.sport) == (self.src, self.sport) else
                                    -len(buf))
            #
            self.src = inet_ntop(AF_INET, self.src)
            self.dst = inet_ntop(AF_INET, self.dst)


        except BaseException as err:
            raise ValueError("[error] %s" % err)
        return self

    @property
    def source(self):
        """(source IP, source port)-tuple of Flow"""
        return (self.src, self.sport)

    @property
    def destination(self):
        """(destination IP, destination port)-tuple of Flow"""
        return (self.dst, self.dport)

    @property
    def time_start(self):
        """Returns start time of Flow"""
        return min(self.timestamps)

    @property
    def time_end(self):
        """Returns end time of Flow"""
        return max(self.timestamps)



    def __len__(self):
        """Return length of Flow in packets."""
        return len(self.lengths)

def clientIp_judge(ip):
    pattern = re.compile(r"^(10\.)|(172\.(1[6-9]|2[0-9]|3[01]))|(192\.168)")
    res = re.match(pattern,ip)
    #匹配成功返回Ture
    if res:
        return True
    else: return False

def natural_sort_key(s):
    # 将文件名按照数字和非数字的部分拆分，然后按照数字部分进行排序
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]
def flow_feature(file_path,tag,fea_path,labeled_num):
    features = []
    f = os.path.isdir(file_path)
    if(f):#如果 file_path 是一个目录，则遍历目录下的所有pcap文件，对每个pcap文件进行流特征提取。如果 file_path 是一个单独的pcap文件，则直接对该文件进行流特征提取
        files = sorted(os.listdir(file_path),key=natural_sort_key)
    else:
        files = [file_path]

    k = 0
    for i in range(len(files)):
        if(f):
            files[i] = file_path + "/" + files[i]

        # print("reading:", files[i])
        f = open(files[i], 'rb')

        pcap = dpkt.pcap.Reader(f)
        flow = Flow()
        start = 0
        # i = 0
        for (ts, buf) in pcap:

            if(start == 0):
                start = ts
            else:
                if(ts - start > 120):
                    break
            # if(i >=5000):break
            # i+=1
            flow.add(ts, buf)
        feature = feature_extract(flow,tag,k,labeled_num)
        k+=1
        features.append(feature)
        f.close()
    features = pd.DataFrame(features)

    if os.path.exists(fea_path):
        features.to_csv(fea_path, mode = 'a',header=False,index=False)
    else:
        features.to_csv(fea_path, header=True, index=False)


def feature_extract(flow:Flow(),tag,k,labeled_num):

        feature = {}
        flow_duration = round(flow.time_end - flow.time_start, 3)

        feature['flow_id'] = str(flow.src) + '_' + str(flow.dst) + '_' + str(flow.sport) + '_' + str(flow.dport) + '_' + str(flow.protocol) + '_' + str(flow.time_start)
        feature['3-tuple'] = str(flow.dst) + '_' + str(flow.dport) + '_' + str(flow.protocol)
        feature['timestamp'] = flow.time_start
        feature['flow_duration'] = flow_duration
        feature['sport'] = flow.sport
        feature['dport'] = flow.dport
        feature['protocol'] = flow.protocol

        # 包长相关特征
        packet_len = np.array(flow.pkt_lengths)
        # 正负向包

        up_packet_len = [packet for packet in packet_len if packet > 0]
        down_packet_len = [abs(packet) for packet in packet_len if packet < 0]
        all_packet_len = [abs(packet) for packet in packet_len]

        # 流数据包总数
        feature['pkt_nums'] = len(all_packet_len)
        # 正向包数
        feature['up_pkt_nums'] = len(up_packet_len)
        # 反向包数
        feature['down_pkt_nums'] = len(down_packet_len)
        # 数据包平均值
        feature['pkt_mean'] = np.mean(all_packet_len)
        # 上行数据包平均值
        if len(up_packet_len) == 0:
            feature['up_pkt_mean'] = 0
        else:
            feature['up_pkt_mean'] = np.mean(up_packet_len)

        # 下行数据包平均值
        if len(down_packet_len) == 0:
            feature['down_pkt_mean'] = 0
        else:
            feature['down_pkt_mean'] = np.mean(down_packet_len)

        # 流最长包
        feature['max_len'] = max(all_packet_len)
        # 正向最长包
        if len(up_packet_len) == 0:
            feature['up_max_len'] = 0
        else:
            feature['up_max_len'] = max(up_packet_len)

        # 反向最长包
        if len(down_packet_len) == 0:
            feature['down_max_len'] = 0
        else:
            feature['down_max_len'] = max(down_packet_len)

        # 流最短包
        feature['min_len'] = min(all_packet_len)

        # 正向最短包
        if len(up_packet_len) == 0:
            feature['up_min_len'] = 0
        else:
            feature['up_min_len'] = min(up_packet_len)

        # 反向最短包
        if len(down_packet_len) == 0:
            feature['down_min_len'] = 0
        else:
            feature['down_min_len'] = min(down_packet_len)

        # 数据包大小标准差
        feature['pkt_standard_deviation'] = round(np.std(all_packet_len), 3)
        # 上行数据包大小标准差
        if (len(up_packet_len) == 0):
            feature['up_pkt_standard_deviation'] = 0
        else:
            feature['up_pkt_standard_deviation'] = round(np.std(up_packet_len), 3)
        # 下行数据包大小标准差
        if (len(down_packet_len) == 0):
            feature['down_pkt_standard_deviation'] = 0
        else:
            feature['down_pkt_standard_deviation'] = round(np.std(down_packet_len), 3)

        # 上下行包数比
        if (len(down_packet_len) == 0):
            feature["pkt_ratio"] = 0
        else:
            feature["pkt_ratio"] = round(len(up_packet_len) / len(down_packet_len),3)

        # 每秒包数
        if flow_duration == 0:
            flow_pkt_s = 0
            up_pkt_s = 0
            down_pkt_s = 0
        else:
            flow_pkt_s = round(len(all_packet_len) / flow_duration, 3)
            up_pkt_s = round(len(up_packet_len) / flow_duration, 3)
            down_pkt_s = round(len(down_packet_len) / flow_duration, 3)

        feature['flow_pkt/s'] = flow_pkt_s
        feature['up_pkt/s'] = up_pkt_s
        feature['down_pkt/s'] = down_pkt_s
        if (down_pkt_s == 0):
            feature["flow_pkt/s_raito"] = 0
        else:
            feature['flow_pkt/s_raito'] = round(up_pkt_s / down_pkt_s,3)

        up_bytes = sum(up_packet_len)
        down_bytes = sum(down_packet_len)
        sum_bytes = sum(all_packet_len)

        # # 流的总字节数
        feature['sum_bytes'] = sum_bytes
        # 正向总字节数
        feature['up_bytes'] = up_bytes
        # 反向总字节数
        feature['down_bytes'] = down_bytes

        # 上下行字节比
        if (down_bytes == 0):
            feature["bytes_ratio"] = 0
        else:
            feature["bytes_ratio"] = round(up_bytes / down_bytes,3)

        # 每秒字节数
        if flow_duration == 0:
            flow_byte_s = 0
            up_byte_s = 0
            down_byte_s = 0
        else:
            flow_byte_s = round(sum_bytes / flow_duration, 3)
            up_byte_s = round(up_bytes / flow_duration, 3)
            down_byte_s = round(down_bytes / flow_duration, 3)
        feature['flow_byte/s'] = flow_byte_s
        feature['up_byte/s'] = up_byte_s
        feature['down_byte/s'] = down_byte_s
        if (down_byte_s == 0):
            feature["flow_byte/s_raito"] = 0
        else:
            feature['flow_byte/s_raito'] = round(up_byte_s / down_byte_s,3)
        time_intervals = []
        up_time_intervals = []
        down_time_intervals = []
        pre_i1 = -1
        pre_i2 = -1
        for i in range(len(flow.timestamps)):
            if (i == 0):
                continue
            else:
                time_intervals.append(flow.timestamps[i] - flow.timestamps[i - 1])

            if (flow.pkt_lengths[i] > 0):

                if (pre_i1 != -1):
                    up_time_intervals.append(round(flow.timestamps[i] - flow.timestamps[pre_i1], 6))
                pre_i1 = i;

            if (flow.pkt_lengths[i] < 0):
                if (pre_i2 != -1):
                    down_time_intervals.append((round(flow.timestamps[i] - flow.timestamps[pre_i2], 6)))
                pre_i2 = i;

        # 时间间隔最大值
        if (len(time_intervals) == 0):
            feature["max_interval"] = 0
        else:
            feature["max_interval"] = max(time_intervals)

        # 上行时间间隔最大值
        if (len(up_time_intervals) == 0):
            feature["up_max_interval"] = 0
        else:
            feature["up_max_interval"] = max(up_time_intervals)

        # 下行时间间隔最大值
        if (len(down_time_intervals) == 0):
            feature["down_max_interval"] = 0
        else:
            feature["down_max_interval"] = max(down_time_intervals)

        # 时间间隔最小值
        if (len(time_intervals) == 0):
            feature["min_interval"] = 0
        else:

            feature["min_interval"] = min(time_intervals)

        # 上行时间间隔最小值
        if (len(up_time_intervals) == 0):
            feature["up_min_interval"] = 0
        else:
            feature["up_min_interval"] = min(up_time_intervals)

        # 下行时间间隔最小值
        if (len(down_time_intervals) == 0):
            feature["down_min_interval"] = 0
        else:
            feature["down_min_interval"] = min(down_time_intervals)

        # 时间间隔平均值
        if(len(time_intervals) == 0):
            feature["interval_mean"] = 0
        else:
            feature["interval_mean"] = np.mean(time_intervals)

        # 上行时间间隔平均值
        if (len(up_time_intervals) == 0):
            feature["up_interval_mean"] = 0
        else:
            feature["up_interval_mean"] = np.mean(up_time_intervals)

        # 下行时间间隔平均值
        if (len(down_time_intervals) == 0):
            feature["down_interval_mean"] = 0
        else:
            feature["down_interval_mean"] = np.mean(down_time_intervals)

        # 时间间隔标准差
        if(len(time_intervals)==0):
            feature["interval_std"] = 0
        else:
            feature["interval_std"] = np.std(time_intervals)

        # 上行时间间隔标准差
        if (len(up_time_intervals) == 0):
            feature["up_interval_std"] = 0
        else:
            feature["up_interval_std"] = np.std(up_time_intervals)

        # 下行时间间隔标准差
        if (len(down_time_intervals) == 0):
            feature["down_interval_std"] = 0
        else:
            feature["down_interval_std"] = np.std(down_time_intervals)

        # 前十个数据包的特征
        if(len(all_packet_len) <= 10):
            n = 2
        else:
            n = 10

        top_10_up_pkt_len = [packet for id, packet in enumerate(packet_len) if packet > 0 and id < n]
        top_10_down_pkt_len = [abs(packet) for id, packet in enumerate(packet_len) if packet < 0 and id < n]
        if(len(flow.timestamps) < n):
            top_10_duration = round(flow.timestamps[-1] - flow.time_start, 3)
        else:
            top_10_duration = round(flow.timestamps[n-1] - flow.time_start, 3)
        top_10_pkt_max = max(all_packet_len[0:n])
        top_10_pkt_min = min(all_packet_len[0:n])
        top_10_pkt_mean = np.mean(all_packet_len[0:n])
        top_10_pkt_std = np.std(all_packet_len[0:n])

        top_10_bytes = sum(all_packet_len[0:n])
        top_10_up_bytes = sum(top_10_up_pkt_len)
        top_10_down_bytes = sum(top_10_down_pkt_len)

        top_10_time_intervals = time_intervals[0:n]
        top_10_time_max_interval = max(top_10_time_intervals)
        top_10_time_min_interval = min(top_10_time_intervals)
        top_10_time_mean_interval = np.mean(top_10_time_intervals)
        top_10_time_interval_std = np.std(top_10_time_intervals)

        feature["top_10_duration"] = top_10_duration
        feature["top_10_pkt_max"] = top_10_pkt_max
        feature["top_10_pkt_min"] = top_10_pkt_min
        feature["top_10_pkt_mean"] = top_10_pkt_mean
        feature["top_10_pkt_std"] = top_10_pkt_std

        feature["top_10_bytes"] = top_10_bytes
        feature["top_10_time_max_interval"] = top_10_time_max_interval
        feature["top_10_time_min_interval"] = top_10_time_min_interval
        feature["top_10_time_mean_interval"] = top_10_time_mean_interval
        feature["top_10_time_interval_std"] = top_10_time_interval_std
        if(top_10_duration == 0):
            feature["top_10_byte/s"] = 0
        else:
            feature["top_10_byte/s"] = np.round(top_10_bytes/top_10_duration,3)
        # 上下行字节比
        if (top_10_down_bytes == 0):
            feature["top_10_bytes_ratio"] = 0
        else:
            feature["top_10_bytes_ratio"] = round(top_10_up_bytes / top_10_down_bytes, 3)

        #数据包分片特征
        up_slice_list = []
        down_slice_list = []
        up_slice = 0
        down_slice = 0
        for pkt in flow.pkt_lengths:
            if(pkt > 0):
                up_slice += 1
                if(down_slice > 0):
                    down_slice_list.append(down_slice)
                    down_slice = 0
            else:
                if(up_slice > 0):
                    up_slice_list.append(up_slice)
                    up_slice = 0
                down_slice+=1

        if up_slice > 0:
            up_slice_list.append(up_slice)
        if down_slice > 0:
            down_slice_list.append(down_slice)
        if(up_slice_list):
            up_slice_max = max(up_slice_list)
            up_slice_min = min(up_slice_list)
            up_slice_mean = np.mean(up_slice_list)
            up_slice_std = np.std(up_slice_list)
        else:
            up_slice_max = 0
            up_slice_min = 0
            up_slice_mean = 0
            up_slice_std = 0

        if(down_slice_list):
            down_slice_max = max(down_slice_list)
            down_slice_min = min(down_slice_list)
            down_slice_mean = np.mean(down_slice_list)
            down_slice_std = np.std(down_slice_list)
        else:
            down_slice_max = 0
            down_slice_min = 0
            down_slice_mean = 0
            down_slice_std = 0

        feature["up_slice_max"] = up_slice_max
        feature["up_slice_min"] = up_slice_min
        feature["up_slice_mean"] = up_slice_mean
        feature["up_slice_std"] = up_slice_std

        feature["down_slice_max"] = down_slice_max
        feature["down_slice_min"] = down_slice_min
        feature["down_slice_mean"] = down_slice_mean
        feature["down_slice_std"] = down_slice_std
        if(k<labeled_num):
            feature["tag"] = tag
        else:
            feature["tag"] = -1
        feature["real_tag"] = tag
        return feature

def has_subdirectories(directory):
    # 遍历目录中的所有项
    for item in os.listdir(directory):
        # 构建完整路径
        full_path = os.path.join(directory, item)
        # 检查这个路径是否是一个目录
        if os.path.isdir(full_path):
            return True
    return False
def exc(file_path,fea_path,labeled_num):
    l = 0
    if(has_subdirectories(file_path)):
        for _, dirs, _ in os.walk(file_path):#dirs 是 file_path 目录下的子目录列表
            for name in dirs:#递归遍历子目录
                flow_feature(file_path +"/" +name,l,fea_path,labeled_num)
                l += 1
    else:
        files = os.listdir(file_path)
        for f in files:
            flow_feature(file_path +"/"+ f, l, fea_path,labeled_num)


if __name__ == '__main__':
    p = "../../../Data"
    labeled_num = [25]#控制每种类型的样本的标记样本数量，例如labeled_num = [5,10,15,20,25,30,35,40,45,50]
    for k in labeled_num:
        fea_path = "./flow_fea" + str(k) + ".csv"
        exc(p,fea_path,k)
