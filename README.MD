# ����Mapreduce��KNNʵ��

## ��Ŀ����

- ����Ŀʵ����KNN�㷨��Hadoopƽ̨���ڵ�MapReduceʵ�֡�

  ʹ�õ����������β�����ݼ����ݼ��ڰ��� 3 �๲ 30 ����¼��ÿ����¼���� 4 ������������ͨ����4������Ԥ���β�������ڣ�iris-setosa, iris-versicolour, iris-virginica���е���һ���

  ѵ����������Ϊ������ֵ1������ֵ2��.......����ǩ

  ���Լ�������Ϊ������ֵ1������ֵ2��.......����ȷ��ǩ

## ����˼·��<br>
   ��1������������������ѵ������֮��ľ��룻<br>
   ��2�����վ���ĵ�����ϵ��������<br>
   ��3��ѡȡ������С��K����(K=3)��<br>
   ��4����ǰK�����г��ִ������������Ϊ�������ݵķ�������<br>

## �ļ����

```
KNN
������ img  //���н����ͼ
������ output  //��Ž������ļ���
|    ������ part-r-00000  //������
������ test //���Լ��ļ���
|    ������ iris_test_data.csv
������ train //���Լ��ļ���
|    ������ iris_train.csv
������ KNN.jar  
������ KNN.java  //Դ�����ļ�
������ README.MD  