Promoter = read.table('fabg1G609A.fasta_output.txt', header=T, sep='\t')
str(Promoter)
attach(Promoter)
#install.packages('ggplot2')
library(ggplot2)
ggplot(Promoter, aes(x=Position, y=Percent.Accuracy)) +  geom_line()
plot = ggplot(Promoter, aes(x=Position, y=Percent.Accuracy)) +  geom_point()+ geom_smooth(span=0.1)
plot

#which(Promoter[,2]>0.80)
#Promoter[,2]
#Promoter[which(Promoter[,2]>0.87)]
