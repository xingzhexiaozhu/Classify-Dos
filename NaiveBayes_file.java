import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.ReflectionUtils;

/**
 * 伯努利模型（Bernoulli model） –以文件为粒度
 * 	类条件概率P(tk|c)=（类c下包含单词tk的文件数+1）/(类c下文件总数+2)
 * 	先验概率P(c)=类c下文件总数/整个训练样本的文件总数
 */

public class NaiveBayesBasedOnFile {
	/*
	 * 在main中为各个任务申请新的conf和job，并设置各个job之间的任务关系
	 */
	static String[] otherArgs;
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();
		otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 5) {
			System.err.println("Usage: NaiveBayesClassification <in> <out>");
			System.exit(5);
		}		
		FileSystem hdfs = FileSystem.get(conf);
		
		/* 判断job1对应输出目录文件是否存在，如果存在则先删除它 */		
		Path path1 = new Path(otherArgs[1]);
		if (hdfs.exists(path1))
			hdfs.delete(path1, true);// 递归删除
		Job job1 = new Job(conf, "ClassDocNums");
		job1.setJarByClass(NaiveBayesBasedOnFile.class);	
		job1.setInputFormatClass(SequenceFileInputFormat.class);
		job1.setOutputFormatClass(SequenceFileOutputFormat.class);
		//设置输入输出格式
		job1.setMapperClass(ClassName_DocNums_Map.class);
		job1.setMapOutputKeyClass(Text.class);//map阶段的输出的key 
		job1.setMapOutputValueClass(IntWritable.class);//map阶段的输出的value 
		job1.setCombinerClass(ClassName_DocNums_Reduce.class);
		job1.setReducerClass(ClassName_DocNums_Reduce.class);
		job1.setOutputKeyClass(Text.class);//reduce阶段的输出的key 
		job1.setOutputValueClass(IntWritable.class);//reduce阶段的输出的value 
		//加入控制容器 
		ControlledJob ctrljob1 = new  ControlledJob(conf);
		ctrljob1.setJob(job1);
		//job1的输入输出文件路径
		FileInputFormat.addInputPath(job1, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job1, new Path(otherArgs[1]));
		
		/* 判断job2对应输出目录文件是否存在，如果存在则先删除它 */
		Path path2 = new Path(otherArgs[2]);
		if(hdfs.exists(path2))
			hdfs.delete(path2, true);
		Job job2 = new Job(conf, "ClassWordDocNums");
		job2.setJarByClass(NaiveBayesBasedOnFile.class);
		job2.setInputFormatClass(SequenceFileInputFormat.class);
		job2.setOutputFormatClass(SequenceFileOutputFormat.class);
		job2.setMapperClass(ClassWord_Docs_Map.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(Text.class);
		job2.setReducerClass(ClassWord_Docs_Reduce.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(IntWritable.class);
		//加入控制容器 
		ControlledJob ctrljob2 = new  ControlledJob(conf);
		ctrljob2.setJob(job2);
		//job1的输入输出文件路径
		FileInputFormat.addInputPath(job2, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2]));
		
		/* 判断job4对应输出目录文件是否存在，如果存在则先删除它 */
		Path path3 = new Path(otherArgs[4]);
		if(hdfs.exists(path3))
			hdfs.delete(path3, true);
		Job job3 = new Job(conf, "TestSet");
		job3.setJarByClass(NaiveBayesBasedOnFile.class);
		job3.setInputFormatClass(SequenceFileInputFormat.class);
		job3.setOutputFormatClass(SequenceFileOutputFormat.class);		
		job3.setMapperClass(Doc_Class_Map.class);
		job3.setMapOutputKeyClass(Text.class);
		job3.setMapOutputValueClass(Text.class);
		job3.setReducerClass(Doc_Class_Reduce.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(Text.class);
		//加入控制容器 
		ControlledJob ctrljob3 = new  ControlledJob(conf);
		ctrljob3.setJob(job3);
		//job4的输入输出文件路径
		FileInputFormat.addInputPath(job3, new Path(otherArgs[3]));
		FileOutputFormat.setOutputPath(job3, new Path(otherArgs[4]));
		
		//作业之间依赖关系
		ctrljob3.addDependingJob(ctrljob1);
		ctrljob3.addDependingJob(ctrljob2);
		
		//主的控制容器，控制上面的子作业 		
		JobControl jobCtrl = new JobControl("NaiveBayes");
		//添加到总的JobControl里，进行控制
		jobCtrl.addJob(ctrljob1);
		jobCtrl.addJob(ctrljob2);
		jobCtrl.addJob(ctrljob3);
		
		//在线程启动，记住一定要有这个
	    Thread  theController = new Thread(jobCtrl); 
	    theController.start(); 
	    while(true){
	        if(jobCtrl.allFinished()){//如果作业成功完成，就打印成功作业的信息 
	        	System.out.println(jobCtrl.getSuccessfulJobList()); 
	        	jobCtrl.stop(); 
	        	break; 
	        }
	    }  
	}

	/*
	 * 第一个MapReduce用于处理序列化的文件,统计每个类对应的文件数量
	 * 为计算先验概率准备:
	 * 输入:args[0],序列化的训练集,key为<类名:文档名>,value为文档中对应的单词.形式为<<ClassName:Doc>,word1 word2...>
	 * 输出:args[1],key为类名,value为类对应的文档数目,即<ClassName,DocNums> 
	 */
	public static class ClassName_DocNums_Map extends Mapper<Text, Text, Text, IntWritable>{
		private Text newKey = new Text();
		private final static IntWritable one = new IntWritable(1);
		public void map(Text key, Text value, Context context) throws IOException, InterruptedException{
			int index = key.toString().indexOf(":");
			newKey.set(key.toString().substring(0, index));
			context.write(newKey, one);
		}
	}
	public static class ClassName_DocNums_Reduce extends Reducer<Text, IntWritable, Text, IntWritable>{
		private IntWritable result = new IntWritable();
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
			int sum = 0;
			for(IntWritable value:values){
				sum += value.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}
	
	/*
	 * 第二个MapReduce用于处理序列化的文件,统计各个类Ci中包含单词tk的文件数
	 * 输入:args[0],序列化的训练集,key为<类名:文档名>,value为文档中对应的单词.形式为<<ClassName:Doc>,word1 word2...>
	 * 输出:args[2],Map阶段输出<<类名:单词>,文档名>,即<<ClassName:word>,Doc>,同时为Reduce阶段的输入
	 *     Rdeuce阶段遍历Value值,如果doc不同则计数加1,最终输出的是每个单词在每个类中多少个文档中出现过
	 */
	public static class ClassWord_Docs_Map extends Mapper<Text, Text, Text, Text>{
		private Text newKey = new Text();
		private Text newValue = new Text();
		public void map(Text key, Text value, Context context) throws IOException, InterruptedException{
			StringTokenizer itr = new StringTokenizer(value.toString());
			while(itr.hasMoreTokens()){
				int index = key.toString().indexOf(":");
				newKey.set(key.toString().substring(0, index) + ":" + itr.nextToken());
				newValue.set(key.toString().substring(index+1, key.toString().length()));
				context.write(newKey, newValue);
			}
		}
	}
	public static class ClassWord_Docs_Reduce extends Reducer<Text, Text, Text, IntWritable>{
		private IntWritable result = new IntWritable();
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			int sum = 0;
			for(Text value:values){
				sum++;
			}
			result.set(sum);
			context.write(key, result);			
		}
	}
	
	/*计算先验概率:
	 * 该静态函数计算每个类的文档在总类中占的比例,即先验概率P(c)=类c下文件总数/整个训练样本的文件总数
	 * 输入:对应第一个MapReduce的输出args[1]
	 * 输出:得到HashMap<String,Double>存放的是<类名,概率>
	 */
	private static HashMap<String, Double> classProbably = new HashMap<String, Double>();//<类别，概率>，即<class,priorProbably>
	public static HashMap<String, Double> GetPriorProbably(/*Configuration conf, String filePath*/) throws IOException{		
		Configuration conf = new Configuration();
		String filePath = otherArgs[1]+"/part-r-00000";
		FileSystem fs = FileSystem.get(URI.create(filePath), conf);
		Path path = new Path(filePath);
		SequenceFile.Reader reader = null;
		double totalDocs = 0;
		try {
			reader = new SequenceFile.Reader(fs, path, conf); //创建Reader对象
			Text key = (Text)ReflectionUtils.newInstance(reader.getKeyClass(), conf);
			IntWritable value = (IntWritable)ReflectionUtils.newInstance(reader.getValueClass(), conf);
			long position = reader.getPosition();//获取当前读取的字节位置。设置标记点，标记文档起始位置，方便后面再回来遍历
			while (reader.next(key, value)) {
				totalDocs += value.get();//得到训练集文档总数
			}
			reader.seek(position);//重置到前面定位的标记点
			while(reader.next(key, value)){
				classProbably.put(key.toString(), value.get()/totalDocs);//各类文档的概率 = 各类文档数目/总文档数目
				//System.out.println(key+":"+value.get()+"/"+totalDocs+"\t"+value.get()/totalDocs);
			}
		}finally {
			IOUtils.closeStream(reader);
		}			
		//验证是否得到先验概率
//		for(Map.Entry<String, Double> entry:classProbably.entrySet()){
//			String mykey = entry.getKey().toString();
//			double myvalue = Double.parseDouble(entry.getValue().toString());
//			System.out.println(mykey + "\t" + myvalue + "\t");
//		}		
		return classProbably;	
	}
	
	/*计算条件概率
	 * 该静态函数计算出各个单词在各个类别中占的概率,即类条件概率P(tk|c)=（类c下包含单词tk的文件数+1）/(类c下文件总数+2)
	 * 输入:第一个MapReduce的输出args[1](各个类的文档总数),第二个MapReduce的输出[2](类C中各个单词在出现的文档总数)
	 * 输出:得到HashMap<String,Double>存放的是<<类名:单词>,概率>
	 */
	private static HashMap<String, Double> wordsProbably = new HashMap<String, Double>();//<<类别:单词>，概率>，即<<class:word>,conditionProbably>
	public static HashMap<String, Double> GetConditionProbably() throws IOException{
		HashMap<String, Double> ClassTotalDocNums = new HashMap<String, Double>();
		Configuration conf = new Configuration();
		String ClassTotalDocNumsPath = otherArgs[1]+"/part-r-00000";
		String ClassWordDocNumsPath = otherArgs[2]+"/part-r-00000";
		
		FileSystem fs1 = FileSystem.get(URI.create(ClassTotalDocNumsPath), conf);
		Path path1 = new Path(ClassTotalDocNumsPath);
		SequenceFile.Reader reader1 = null;
		try{
			reader1 = new SequenceFile.Reader(fs1, path1, conf);//创建Reader对象
			Text key1 = (Text)ReflectionUtils.newInstance(reader1.getKeyClass(), conf);
			IntWritable value1 = (IntWritable)ReflectionUtils.newInstance(reader1.getValueClass(), conf);
			while(reader1.next(key1, value1)){				
				ClassTotalDocNums.put(key1.toString(), value1.get()*1.0);//得到每个类及类对应的文件数目
				//System.out.println(key1.toString() + "\t" + ClassTotalDocNums.get(key1.toString()));	
			}
		}finally{
			IOUtils.closeStream(reader1);
		}
		//验证是否得到类及类单词总数
//		for(Map.Entry<String, Double> entry:ClassTotalDocNums.entrySet()){
//			String mykey = entry.getKey();
//			double myvalue = entry.getValue();
//			System.out.println(mykey + "\t" + myvalue);
//		}
		
		FileSystem fs2 = FileSystem.get(URI.create(ClassWordDocNumsPath), conf);
		Path path2 = new Path(ClassWordDocNumsPath);
		SequenceFile.Reader reader2 = null;
		try{
			reader2 = new SequenceFile.Reader(fs2, path2, conf);
			Text key2 = (Text)ReflectionUtils.newInstance(reader2.getKeyClass(), conf);
			IntWritable value2 = (IntWritable)ReflectionUtils.newInstance(reader2.getValueClass(), conf);
			Text newKey = new Text();
			while(reader2.next(key2, value2)){				
				int index = key2.toString().indexOf(":");
				newKey.set(key2.toString().substring(0, index));//得到单词所在类的类名,以便后面根据类名查找对应类的文件数目
				wordsProbably.put(key2.toString(), (value2.get()+1)/(ClassTotalDocNums.get(newKey.toString())+2));
				//System.out.println(key2.toString() + "\t" + (value2.get()+1)/(ClassNameTotalCounts.get(newKey.toString())+2));
			}
			//对于同一个类别没有出现过的单词的概率一样，1/(ClassTotalDocNums.get(newKey.toString())+2)
			//遍历类，每个类别中再加一个没有出现单词的概率，其格式为<class,probably>
			for(Map.Entry<String,Double> entry:ClassTotalDocNums.entrySet()){
				wordsProbably.put(entry.getKey().toString(), 1.0/(ClassTotalDocNums.get(entry.getKey().toString()) + 2));
				//System.out.println(entry.getKey().toString() + "\t" + 1.0/(ClassTotalDocNums.get(entry.getKey().toString())+2));
			}
		}finally{
			IOUtils.closeStream(reader2);
		}
		//验证是否得到条件概率
//		for(Map.Entry<String, Double> entry:wordsProbably.entrySet()){
//			String mykey = entry.getKey();
//			double myvalue = entry.getValue();
//			System.out.println(mykey + "\t" + myvalue);
//		}		
		return wordsProbably;		
	}
	
	/*
	 * 第三个MapReduce处理测试集中的文档,得到key为<doc,word>,value计数(每份文档中的各个单词出现次数)
	 * 输入:args[3],序列化的测试集,key为<类名:文档名>,value为文档中对应的单词.形式为<<ClassName:Doc>,word1 word2...>
	 *		HashMap<String,Double> classProbably先验概率
     *      HashMap<String,Double> wordsProbably条件概率
	 * 输出:args[4],输出:计算结果<doc,class>,即文档及对应的类别
	 */
	public static class Doc_Class_Map extends Mapper<Text, Text, Text, Text>{
		public void setup(Context context)throws IOException{
			GetPriorProbably();
			GetConditionProbably();
		}
		
		private Text newKey = new Text();
		private Text newValue = new Text();
		public void map(Text key, Text value, Context context) throws IOException, InterruptedException{
			int index = key.toString().indexOf(":");
			String docID = key.toString().substring(index + 1, key.toString().length());
			
			for(Map.Entry<String, Double> entry:classProbably.entrySet()){//外层循环遍历所有类别
				String mykey = entry.getKey();
				newKey.set(docID);//新的键值的key为<文档名>
				double tempValue = Math.log(entry.getValue());//构建临时键值对的value为各概率相乘,转化为各概率取对数再相加				
				StringTokenizer itr = new StringTokenizer(value.toString());				
				while(itr.hasMoreTokens()){//内层循环遍历一份测试文档中的所有单词				
					String tempkey = mykey + ":" + itr.nextToken();//构建临时键值对<class:word>,在wordsProbably表中查找对应的概率						
					if(wordsProbably.containsKey(tempkey)){
						//如果测试文档的单词在训练集中出现过，则直接加上之前计算的概率
						tempValue += Math.log(wordsProbably.get(tempkey));
					}else{//如果测试文档中出现了新单词则加上之前计算新单词概率
						tempValue += Math.log(wordsProbably.get(mykey));						
					}
				}
				newValue.set(mykey + ":" + tempValue);///新的value为<类名:概率>,即<class:probably>
				context.write(newKey, newValue);//一份文档遍历在一个类中遍历完毕,则将结果写入文件,即<docID,<class:probably>>
				//System.out.println(newKey + "\t" + newValue);
			}
		}
	}
	public static class Doc_Class_Reduce extends Reducer<Text, Text, Text, Text>{
		Text newValue = new Text();	
		public void reduce(Text key, Iterable<Text> values,Context context)throws IOException, InterruptedException {
			boolean flag = false;//标记,若第一次循环则先赋值,否则比较若概率更大则更新
			String tempClass = null;
			double tempProbably = 0.0;
	        for (Text value : values) {//对于每份文档找出最大的概率所对应的类 
	        	int index = value.toString().indexOf(":");	        	
        		if(flag != true){
	        		tempClass = value.toString().substring(0, index);
	        		tempProbably = Double.parseDouble(value.toString().substring(index+1, value.toString().length()));	        		
	        		flag = true;
	        	}else{
	        		if(Double.parseDouble(value.toString().substring(index+1, value.toString().length())) > tempProbably)
	        			tempClass = value.toString().substring(0, index);
	        			//tempProbably = Double.parseDouble(value.toString().substring(index+1, value.toString().length()));	
	        	}
        	}	        	
	        
        	newValue.set(tempClass);      	
	        context.write(key, newValue);	
	        System.out.println(key + "\t" + newValue);
	    }
	}	
}
