import java.io.IOException;  
import java.util.StringTokenizer;  
import org.apache.hadoop.conf.Configuration;  
import org.apache.hadoop.fs.Path;  
import org.apache.hadoop.io.Text;  
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.Job;  
import org.apache.hadoop.mapreduce.Mapper;  
import org.apache.hadoop.mapreduce.Reducer;  
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;  
import org.apache.hadoop.mapreduce.lib.input.FileSplit;  
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;  
import org.apache.hadoop.util.GenericOptionsParser;  
  
import java.util.Collections;
import java.util.Comparator;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

public class KNN{
    public static void main(String[] args) throws Exception{
        // 创建配置对象
        Configuration conf=new Configuration();

        if(args.length!=2)  
        {  
            System.out.println("Input Error! Usage: KNN <in> <out>");  
            System.exit(2);  
        }
        
        Path inputPath=new Path(args[0]);  
        Path outputPath=new Path(args[1]); 
        outputPath.getFileSystem(conf).delete(outputPath, true);
        // 创建Job对象
        Job job=Job.getInstance(conf, "KNN");  
        // 设置运行Job的类
        job.setJarByClass(KNN.class);
        // 设置Mapper类
        job.setMapperClass(KnnMapper.class);  
        job.setMapOutputKeyClass(Text.class);  
        job.setMapOutputValueClass(Text.class);  
         // 设置Combine类 
        job.setCombinerClass(KnnCombiner.class);  
        // 设置Reducer类
        job.setReducerClass(KnnReducer.class);          
        job.setOutputKeyClass(Text.class);  
        job.setOutputValueClass(Text.class);  
        // 设置输入输出的路径
        FileInputFormat.addInputPath(job, inputPath);  
        FileOutputFormat.setOutputPath(job, outputPath);  
        
        // 提交job
        boolean b = job.waitForCompletion(true);
        if(!b) System.out.println("KNN task fail!");
    }

    public static class KnnMapper extends Mapper<Object, Text, Text, Text>{
        private ArrayList<ArrayList<Float>> test = new ArrayList<ArrayList<Float>> ();
    
        @Override  
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException{  
            String[] s = value.toString().split(",");
            String label = s[s.length - 1];        
            for (int i=0; i<test.size(); i++){
                ArrayList<Float> curr_test = test.get(i);
                double tmp = 0;
                for(int j=0; j<curr_test.size(); j++){
                    tmp += (curr_test.get(j) - Float.parseFloat(s[j]))*(curr_test.get(j) - Float.parseFloat(s[j]));
                }
                context.write(new Text(Integer.toString(i)), new Text(Double.toString(tmp)+"&"+label));                
            }

        }
        protected void setup(org.apache.hadoop.mapreduce.Mapper<Object, Text, Text, Text>.Context context) throws java.io.IOException, InterruptedException {
            // load the test vectors
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(context.getConfiguration().get(
                    "org.niubility.learning.test", "./test/iris_test_data.csv")))));
            String line = br.readLine();
            int count = 0;
            while (line != null) {
                String[] s = line.split(",");
                ArrayList<Float> testcase = new ArrayList<Float>();
                for (int i = 0; i < s.length-1; i++){
                    testcase.add(Float.parseFloat(s[i]));
                }
                test.add(testcase);
                line = br.readLine();
                count++;
            }
            br.close();
        }  
    }

    public static class KnnCombiner extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            ArrayList<Distance_Label> dis_Label_set = new ArrayList<Distance_Label>();
            for (Text value : values){
                String[] s = value.toString().split("&");
                Distance_Label tmp = new Distance_Label();
                tmp.label = s[1];
                tmp.distance = Float.parseFloat(s[0]);
                dis_Label_set.add(tmp);
            }
            //排序 
            Collections.sort(dis_Label_set, new Comparator<Distance_Label>(){
                @Override
                public int compare(Distance_Label a, Distance_Label b){ 
                    if (a.distance > b.distance){
                        return 1; //对距离进行排序，距离越小，排序越前
                    }
                    return -1;
                }
            });

            final int k = 3; //K值

            //统计前K个最近样例的标签 KNN
            for (int i=0; i<dis_Label_set.size() && i<k; i++){
               context.write(key, new Text(dis_Label_set.get(i).label));
            }
        
        }  
    }

    public static class KnnReducer extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            HashMap<String, Integer> res = new HashMap<String, Integer>();
            for(Text val:values)  
            {  
                if (!res.containsKey(val)){
                    res.put(val.toString(), 1);
                }
                else res.put(val.toString(), res.get(val.toString())+1); 
            }  
            //获取次数最多的标签
            int max = 0;
            String resLabel = "";
            for (String label:res.keySet()){
                if (max < res.get(label)){
                    max = res.get(label);
                    resLabel = label;
                }
            }   
            context.write(key, new Text(resLabel));  
        }
    }

    public static class Distance_Label{
        public float distance;//距离
        public String label;//标签
    }  
}

