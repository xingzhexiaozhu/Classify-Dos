import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

/**
 * SequenceFileWriter:
 * 序列化小文件，格式为<<dir:doc>,contents>
 * eg:I01002:484619newsML.txt	national oilseed processorsassociation weekly soybean crushings reporting members...
 */

public class SequenceFileWriter {

	/**
	 * 该函数作用读取文件内容,一个参数:
	 * 输入：文件名(file)
	 * 输出：文件内容合并成的字符串(result)
	 */	
	private static String fileToString(File file) throws IOException{
		BufferedReader reader = new BufferedReader(new FileReader(file));		
		
		String line = null;
		String result = "";
		while((line = reader.readLine()) != null){
			if(line.matches("[a-zA-Z]+")){//过滤掉以数字开头的词
				result += line + " ";//单词之间以空格符隔开
				//System.out.println(line);
			}
		}
		reader.close();
		return result;
	}	
	
	/**
	 * 将一个文件夹下的所有文件序列化,两个参数:
     * 输入：args[0]：准备序列化的文件的路径
     * 输出：args[1]：序列化后准备输出的文件路径名字
     */
	public static void main(String[] args) throws IOException {
		File[] dirs = new File(args[0]).listFiles();
		
		String uri = args[1];
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(URI.create(uri), conf);
		Path path = new Path(uri);
		
		Text key = new Text();
		Text value = new Text();
		
		SequenceFile.Writer writer = null;
		try{
			writer = SequenceFile.createWriter(fs, conf, path, key.getClass(), value.getClass());
		
			for(File dir:dirs){
				File[] files = dir.listFiles();
				for(File file:files){
					//key：目录名+":"+文件名					
					key.set(dir.getName() + ":" + file.getName());
					//value：文件内容
					value.set(fileToString(file));
					writer.append(key, value);
					//System.out.println(key + "\t" + value);
				}
			}
		}finally{
			IOUtils.closeStream(writer);
		}		
	}

}
