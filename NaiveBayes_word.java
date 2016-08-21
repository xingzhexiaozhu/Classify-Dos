import java.io.IOException;
import java.net.URI;
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
 * 多项式模型( multinomial model)  –以单词为粒度
 * 	类条件概率P(tk|c)=(类c下单词tk在各个文档中出现过的次数之和+1)/（类c下单词总数+训练样本中不重复特征词总数）
 * 	先验概率P(c)=类c下的单词总数/整个训练样本的单词总数
 */

public class NaiveBayesBasedOnWord {
	static String[] otherArgs; 
	public static void main(String[] args) throws IOException {
		Configuration conf = new Configuration();
		otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if(otherArgs.length != 6){
			System.err.println("Usage: NaiveBayesClassification!");
			System.exit(6);
		}
		
		FileSystem hdfs = FileSystem.get(conf);
		
		Path path1 = new Path(otherArgs[1]);
		if(hdfs.exists(path1))
			hdfs.delete(path1, true);
		Job job1 = new Job(conf, "WordCounts");
		job1.setJarByClass(NaiveBayesBasedOnWord.class);
		job1.setInputFormatClass(SequenceFileInputFormat.class);
		job1.setOutputFormatClass(SequenceFileOutputFormat.class);
		job1.setMapperClass(ClassWordCountsMap.class);
		job1.setMapOutputKeyClass(Text.class);//map阶段的输出的key 
		job1.setMapOutputValueClass(IntWritable.class);//map阶段的输出的value 
//		job1.setCombinerClass(WordCountsReduce.class);
		job1.setReducerClass(ClassWordCountsReduce.class);
		job1.setOutputKeyClass(Text.class);//reduce阶段的输出的key 
		job1.setOutputValueClass(IntWritable.class);//reduce阶段的输出的value 
		//加入控制容器 
		ControlledJob ctrljob1 = new  ControlledJob(conf);
		ctrljob1.setJob(job1);
		//job1的输入输出文件路径
		FileInputFormat.addInputPath(job1, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job1, path1);
		
		Path path2 = new Path(otherArgs[2]);
		if(hdfs.exists(path2))
			hdfs.delete(path2, true);
		Job job2 = new Job(conf, "ClassTotalWords");
		job2.setJarByClass(NaiveBayesBasedOnWord.class);
		job2.setInputFormatClass(SequenceFileInputFormat.class);
		job2.setOutputFormatClass(SequenceFileOutputFormat.class);
		job2.setMapperClass(ClassTotalWordsMap.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(IntWritable.class);
//		job2.setCombinerClass(ClassTotalWordsReduce.class);
		job2.setReducerClass(ClassTotalWordsReduce.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(IntWritable.class);
		//加入控制容器 
		ControlledJob ctrljob2 = new ControlledJob(conf);
		ctrljob2.setJob(job2);
		//job2的输入输出文件路径
		FileInputFormat.addInputPath(job2, new Path(otherArgs[1] + "/part-r-00000"));
		FileOutputFormat.setOutputPath(job2, path2);
		
		Path path3 = new Path(otherArgs[3]);
		if(hdfs.exists(path3))
			hdfs.delete(path3, true);
		Job job3 = new Job(conf, "DiffTotalWords");
		job3.setJarByClass(NaiveBayesBasedOnWord.class);
		job3.setInputFormatClass(SequenceFileInputFormat.class);
		job3.setOutputFormatClass(SequenceFileOutputFormat.class);
		job3.setMapperClass(DiffTotalWordsMap.class);
		job3.setMapOutputKeyClass(Text.class);
		job3.setMapOutputValueClass(IntWritable.class);
//		job3.setCombinerClass(ClassDiffTotalWordsReduce.class);
		job3.setReducerClass(DiffTotalWordsReduce.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(IntWritable.class);
		//加入控制容器 
		ControlledJob ctrljob3 = new ControlledJob(conf);
		ctrljob3.setJob(job3);
		//job3的输入输出文件路径
		FileInputFormat.addInputPath(job3, new Path(otherArgs[1] + "/part-r-00000"));
		FileOutputFormat.setOutputPath(job3, path3);
		
		Path path4 = new Path(otherArgs[5]);
		if(hdfs.exists(path4))
			hdfs.delete(path4, true);
		Job job4 = new Job(conf, "Doc-Class");
		job4.setJarByClass(NaiveBayesBasedOnWord.class);
		job4.setInputFormatClass(SequenceFileInputFormat.class);
		job4.setOutputFormatClass(SequenceFileOutputFormat.class);
		job4.setMapperClass(DocOfClassMap.class);
		job4.setMapOutputKeyClass(Text.class);
		job4.setMapOutputValueClass(Text.class);
		job4.setReducerClass(DocOfClassReduce.class);
		job4.setOutputKeyClass(Text.class);
		job4.setOutputValueClass(Text.class);
		//加入控制容器 
		ControlledJob ctrljob4 = new ControlledJob(conf);
		ctrljob4.setJob(job4);
		//job4的输入输出文件路径
		FileInputFormat.addInputPath(job4, new Path(otherArgs[4]));
		FileOutputFormat.setOutputPath(job4, path4);
		
		//作业之间依赖关系
		ctrljob2.addDependingJob(ctrljob1);
		ctrljob3.addDependingJob(ctrljob1);
		ctrljob4.addDependingJob(ctrljob3);
		ctrljob4.addDependingJob(ctrljob2);
		
		//主的控制容器，控制上面的子作业 		
		JobControl jobCtrl = new JobControl("NaiveBayes");
		//添加到总的JobControl里，进行控制
		jobCtrl.addJob(ctrljob1);
		jobCtrl.addJob(ctrljob2);
		jobCtrl.addJob(ctrljob3);
		jobCtrl.addJob(ctrljob4);
		
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
	 * 第一个MapReduce用于处理序列化的文件，得到<<类名:单词>,单词出现次数>,即<<Class:word>,TotalCounts>
	 * 输入:args[0],序列化的训练集,key为<类名:文档名>,value为文档中对应的单词.形式为<<ClassName:Doc>,word1 word2...>
	 * 输出:args[1],key为<类名:单词>,value为单词出现次数,即<<Class:word>,TotalCounts>
	 */
	public static class ClassWordCountsMap extends Mapper<Text, Text, Text, IntWritable>{
		private Text newKey = new Text();
		private final IntWritable one = new IntWritable(1);
		public void map(Text key, Text value, Context context) throws IOException, InterruptedException{		
			int index = key.toString().indexOf(":");//训练集key=ClassName:DocID
			String Class = key.toString().substring(0, index);
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				newKey.set(Class + ":" + itr.nextToken());//设置新键值key为<类名:单词>,value计数(统计各个类中各个单词的数量)
				context.write(newKey, one);
			}
		}		
	}
	public static class ClassWordCountsReduce extends Reducer<Text, IntWritable, Text, IntWritable>{
		private IntWritable result = new IntWritable();
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
			int sum = 0;
			for(IntWritable value:values){
				sum += value.get();
			}
			result.set(sum);
			context.write(key, result);
			//System.out.println(key + "\t" + result);
		}
	}
	
	/*
	 * 第二个MapReduce在第一个MapReduce计算的基础上进一步得到每个类的单词总数<class,TotalWords>
	 * 输入:args[1],输入格式为<<class,word>,counts>
	 * 输出:args[2],输出key为类名,value为单词总数.格式为<class,Totalwords>
	 */
	public static class ClassTotalWordsMap extends Mapper<Text, IntWritable, Text, IntWritable>{
		private Text newKey = new Text();
		public void map(Text key, IntWritable value, Context context) throws IOException, InterruptedException{
			int index = key.toString().indexOf(":");
			newKey.set(key.toString().substring(0, index));
			context.write(newKey, value);
		}
	}
	public static class ClassTotalWordsReduce extends Reducer<Text, IntWritable, Text, IntWritable>{
		private IntWritable result = new IntWritable();
	    public void reduce(Text key, Iterable<IntWritable> values,Context context)throws IOException, InterruptedException {
	        int sum = 0;
	        for (IntWritable value : values) {            	
	            sum += value.get();
	        }
	        result.set(sum);            
	        context.write(key, result); 
	        //System.out.println(key +"\t"+ result);
	    }
	}
	
	/*
	 * 第三个MapReduce在第一个MapReduce的计算基础上得到训练集中不重复的单词<word,one>
	 * 输入:args[1],输入格式为<<class,word>,counts>
	 * 输出:args[3],输出key为不重复单词,value为1.格式为<word,one>
	 */
	public static class DiffTotalWordsMap extends Mapper<Text, IntWritable, Text, IntWritable>{
		private Text newKey = new Text();		
		public void map(Text key, IntWritable value, Context context) throws IOException, InterruptedException{
			int index = key.toString().indexOf(":");
			newKey.set(key.toString().substring(index+1, key.toString().length()));//设置新键值key为<word>
			context.write(newKey, value);
		}
	}
	public static class DiffTotalWordsReduce extends Reducer<Text, IntWritable, Text, IntWritable>{
		private final IntWritable one = new IntWritable(1);
	    public void reduce(Text key, Iterable<IntWritable> values,Context context)throws IOException, InterruptedException {	        
	        context.write(key, one); 
	        //System.out.println(key +"\t"+ one);
	    }
	}
	
	/* 计算先验概率
	 * 先验概率P(c)=类c下的单词总数/整个训练样本的单词总数
	 * 输入:对应第二个MapReduce的输出,格式为<class,totalWords>
	 * 输出:得到HashMap<String,Double>,即<类名,概率>
	 */
	private static HashMap<String,Double> classProbably = new HashMap<String,Double>();
	public static HashMap<String,Double> GetPriorProbably() throws IOException{
		Configuration conf = new Configuration();			
		String filePath = otherArgs[2]+"/part-r-00000";
		FileSystem fs = FileSystem.get(URI.create(filePath), conf);
		Path path = new Path(filePath);
		SequenceFile.Reader reader = null;
		double totalWords = 0;
		try{
			reader = new SequenceFile.Reader(fs, path, conf);
			Text key = (Text)ReflectionUtils.newInstance(reader.getKeyClass(), conf);
			IntWritable value = (IntWritable)ReflectionUtils.newInstance(reader.getValueClass(), conf);
			long position = reader.getPosition();//设置标记点，标记文档起始位置，方便后面再回来遍历
			while(reader.next(key,value)){
				totalWords += value.get();//得到训练集总单词数
			}
			
			reader.seek(position);//重置到前面定位的标记点
			while(reader.next(key,value)){
				classProbably.put(key.toString(), value.get()/totalWords);//P(c)=类c下的单词总数/整个训练样本的单词总数
				//System.out.println(key+":"+value.get()+"/"+totalWords+"\t"+value.get()/totalWords);
			}
		}finally{
			IOUtils.closeStream(reader);
		}
		return classProbably;		
		
	}
	
	/* 计算条件概率
	 * 条件概率P(tk|c)=(类c下单词tk在各个文档中出现过的次数之和+1)/（类c下单词总数+训练样本中不重复特征词总数）
	 * 输入:对应第一个MapReduce的输出<<class,word>,counts>,第二个MapReduce的输出<class,totalWords>,第三个MapReduce的输出<class,diffTotalWords>
	 * 输出:得到HashMap<String,Double>,即<<类名:单词>,概率>
	 */
	private static HashMap<String, Double> wordsProbably = new HashMap<String, Double>();
	public static HashMap<String, Double> GetConditionProbably() throws IOException{
		Configuration conf = new Configuration();		
		HashMap<String, Double> ClassTotalWords = new HashMap<String, Double>();//每个类及类对应的单词总数
		
		String ClassTotalWordsPath = otherArgs[2]+"/part-r-00000";
		String DiffTotalWordsPath = otherArgs[3]+"/part-r-00000";
		String ClasswordcountsPath = otherArgs[1]+"/part-r-00000";
		double TotalDiffWords = 0.0;
		
		FileSystem fs1 = FileSystem.get(URI.create(ClassTotalWordsPath), conf);
		Path path1 = new Path(ClassTotalWordsPath);
		SequenceFile.Reader reader1 = null;
		try{
			reader1 = new SequenceFile.Reader(fs1, path1, conf);
			Text key1 = (Text)ReflectionUtils.newInstance(reader1.getKeyClass(), conf);
			IntWritable value1 = (IntWritable)ReflectionUtils.newInstance(reader1.getValueClass(), conf);
			while(reader1.next(key1,value1)){
				ClassTotalWords.put(key1.toString(), value1.get()*1.0);
				//System.out.println(key1.toString() + "\t" + value1.get());
			}
		}finally{
			IOUtils.closeStream(reader1);
		}
		
		FileSystem fs2 = FileSystem.get(URI.create(DiffTotalWordsPath), conf);
		Path path2 = new Path(DiffTotalWordsPath);
		SequenceFile.Reader reader2 = null;
		try{
			reader2 = new SequenceFile.Reader(fs2, path2, conf);
			Text key2 = (Text)ReflectionUtils.newInstance(reader2.getKeyClass(), conf);
			IntWritable value2 = (IntWritable)ReflectionUtils.newInstance(reader2.getValueClass(), conf);
			while(reader2.next(key2,value2)){
				TotalDiffWords += value2.get();
			}	
			System.out.println(TotalDiffWords);
		}finally{
			IOUtils.closeStream(reader2);
		}
		
		FileSystem fs3 = FileSystem.get(URI.create(ClasswordcountsPath), conf);
		Path path3 = new Path(ClasswordcountsPath);
		SequenceFile.Reader reader3 = null;
		try{
			reader3 = new SequenceFile.Reader(fs3, path3, conf);
			Text key3 = (Text)ReflectionUtils.newInstance(reader3.getKeyClass(), conf);
			IntWritable value3 = (IntWritable)ReflectionUtils.newInstance(reader3.getValueClass(), conf);
			Text newKey = new Text();
			while(reader3.next(key3,value3)){
				int index = key3.toString().indexOf(":");
				newKey.set(key3.toString().substring(0, index));//得到单词所在的类
				//wordsProbably.put(key3.toString(), (value3.get()+1)/(ClassTotalWords.get(newKey.toString())+ClassDiffTotalWords.get(newKey.toString())));
				wordsProbably.put(key3.toString(), (value3.get()+1)/(ClassTotalWords.get(newKey.toString())+TotalDiffWords));
				                  //<<class:word>,wordcounts/(classTotalNums+TotalDiffWords)>
				//System.out.println(key3.toString() + " \t" + (value3.get()+1) + "/" + (ClassTotalWords.get(newKey.toString())+ "+" +TotalDiffWords));
			}
			//对于同一个类别没有出现过的单词的概率一样，1/(ClassTotalWords.get(class) + TotalDiffWords)
			//遍历类，每个类别中再加一个没有出现单词的概率，其格式为<class,probably>
			for(Map.Entry<String,Double> entry:ClassTotalWords.entrySet()){
				wordsProbably.put(entry.getKey().toString(), 1.0/(ClassTotalWords.get(entry.getKey().toString()) + TotalDiffWords));
				//System.out.println(entry.getKey().toString() +"\t"+ 1.0+"/"+(ClassTotalWords.get(entry.getKey().toString()) +"+"+ TotalDiffWords));
			}
		}finally{
			IOUtils.closeStream(reader3);
		}
//		for(Map.Entry<String, Double> entry:wordsProbably.entrySet()){
//			for(Map.Entry<String, Double> entry1:ClassDiffTotalWords.entrySet()){
//				if(entry.getKey().toString().equals(entry1.getKey().toString()))
//					System.out.println(entry.getKey() + "\t" + entry.getValue().toString());
//			}
//		}	
		return wordsProbably;
	}
	
	/*
	 * 第四个MapReduce进行贝叶斯测试
	 * 输入:args[4],测试文件的路径,测试数据格式<<class:doc>,word1 word2 ...>
	 *      HashMap<String,Double> classProbably先验概率
     *      HashMap<String,Double> wordsProbably条件概率
	 * 输出:args[5],输出每一份文档经贝叶斯分类后所对应的类,格式为<doc,class> 
	 */
	public static class DocOfClassMap extends Mapper<Text, Text, Text, Text>{
		public void setup(Context context)throws IOException{
			GetPriorProbably();
			GetConditionProbably();
		}
		
		private Text newKey = new Text();
		private Text newValue = new Text();
		public void map(Text key, Text value, Context context) throws IOException, InterruptedException{
			int index = key.toString().indexOf(":");
			String docID = key.toString().substring(index+1, key.toString().length());
			for(Map.Entry<String, Double> entry:classProbably.entrySet()){//外层循环遍历所有类别
				String mykey = entry.getKey();//类名
				newKey.set(docID);//新的键值的key为<文档名>
				double tempvalue = Math.log(entry.getValue());//构建临时键值对的value为各概率相乘,转化为各概率取对数再相加
				StringTokenizer itr = new StringTokenizer(value.toString());
				while(itr.hasMoreTokens()){//内层循环遍历一份测试文档中的所有单词	
					String tempkey = mykey + ":" + itr.nextToken();//构建临时键值对<class:word>,在wordsProbably表中查找对应的概率
					if(wordsProbably.containsKey(tempkey)){
						//如果测试文档的单词在训练集中出现过，则直接加上之前计算的概率
						tempvalue += Math.log(wordsProbably.get(tempkey));
					}
					else{//如果测试文档中出现了新单词则加上之前计算新单词概率
						tempvalue += Math.log(wordsProbably.get(mykey));						
					}
				}
				newValue.set(mykey + ":" + tempvalue);//新的键值的value为<类名:概率>,即<class:probably>
				context.write(newKey, newValue);//一份文档遍历在一个类中遍历完毕,则将结果写入文件,即<docID,<class:probably>>
				//System.out.println(newKey + "\t" +newValue);
			}
		}
	}
	public static class DocOfClassReduce extends Reducer<Text, Text, Text, Text>{
		Text newValue = new Text();
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			boolean flag = false;//标记,若第一次循环则先赋值,否则比较若概率更大则更新
			String tempClass = null;
			double tempProbably = 0.0;
			for(Text value:values){
				int index = value.toString().indexOf(":");
				if(flag != true){//循环第一次
					tempClass = value.toString().substring(0, index);
					tempProbably = Double.parseDouble(value.toString().substring(index+1, value.toString().length()));
					flag = true;
				}else{//否则当概率更大时就更新tempClass和tempProbably
					if(Double.parseDouble(value.toString().substring(index+1, value.toString().length())) > tempProbably){
						tempClass = value.toString().substring(0, index);
						tempProbably = Double.parseDouble(value.toString().substring(index+1, value.toString().length()));
					}
				}
			}				
			
			newValue.set(tempClass);
			context.write(key, newValue);
			System.out.println(key + "\t" + newValue);
		}
	}
}
