# Common Commands

## tar

```bash
tar [options][archive-file] [file or dir to be archived]
```

**-c**: This creates an archive file.

**-x**: The option extracts the archive file.

**-f**: Specifies the filename of the archive file.

**-v**: This prints verbose information for any tar operation on the terminal.

**-t**: This lists all the files inside an archive file.

**-u**: This archives a file and then adds it to an existing archive file.

**-A:** This option is used for concatenating the archive files.

**-r**: This updates a file or directory located inside a .tar file

**-z**: Creates a tar file using gzip compression

**-j**: Create an archive file using the bzip2 compression

**-C:** This option is used to untar any file in our current directory or inside the specified directory

**-W**: The -w option verifies an archive file.

常用选项说明：

* 创建压缩文件 -c，解压文件 -x

* 一般都会把 -f -v 给带上，一方面需要借助 -f 指定压缩文件的名字，另一方面借助 -v 将压缩或解压的信息输出到控制台

* 指定压缩类型用 -z 或 -j：

  * -z 为 gzip 压缩，压缩的文件后缀为 `.gz`，无论是创建还是解压 `.gz` 文件都要带上 -z 选项
  * -j 为 bzip2 压缩，压缩的文件后缀为 `.tbz`，无论是创建还是解压 `.tbz` 文件都要带上 -j 选项

  如果只是 `.tar` 文件，可以不用 -z 或 -j 选项

**创建压缩文件示例：**

```bash
tar cvf xxx.tar [file1, file2, ... | directory]
tar cvfz xxx.tar.gz [file1, file2, ... | directory]
tar cvfj xxx.tar.tbz [file1, file2, ... | directory]
```

**解压文件示例：**

```bash
tar xvf xxx.tar (-C specified path)
tar xvfz xxx.tar.gz (-C specified path)
tar xvfj xxx.tar.tbz (-C specified path)
```

