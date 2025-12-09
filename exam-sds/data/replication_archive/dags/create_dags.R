setwd("__your_path_here__") 

rm(list=ls())

library(dagitty)
library(ggdag) 
library(grid)
library(gridExtra)
library(ggplot2)


## FIGURE 1

coords <- list(
  x = c(Y=2, T=0, U=-1, F=1, X=0, T1=-1, T2=0),
  y = c(Y=1, T=1, U=1, F=2, X=0, T1=-1, T2=-1)
)

dag1a <- dagify(Y ~ T + F + X,
              F ~ U,
              T ~ U + X,
              coords = coords)
print( adjustmentSets( dag1a, "T", "Y" ) )


g1atest <- ggdag(dag1a, node_size=8, text_size = 4) + 
  theme_classic() + remove_axes() + theme(axis.line=element_blank()) + ylim(-1.5, 2.5) + xlim(-1.5,2.5) +
  geom_text(x=-1, y=2, label="1(a)", size=4) +
  geom_text(x=0, y=-1, label="Controlling for F and X identifies the effect of T", size=3) +  geom_text(x=-1, y=-.5, label=expression(t[1]), size=3) +
  geom_text(x=0, y=-.5, label=expression(t[2]), size=3) +
  geom_text(x=1, y=-.5, label=expression(t[3]), size=3) +
  geom_text(x=2, y=-.5, label=expression(t[4]), size=3)




coords <- list(
  x = c(Y=2, T=0, U=-1, F=1, X=0, T1=-1, T2=0),
  y = c(Y=1, T=1, U=1, F=2, X=0, T1=-1, T2=-1)
)

dag1b <- dagify(Y ~ T + F + X,
              F ~ T,
              T ~ U + X,
              coords = coords)
print( adjustmentSets( dag1b, "T", "Y" ) )


g1btest <- ggdag(dag1b, node_size=8, text_size = 4) + 
  theme_classic() + remove_axes() + theme(axis.line=element_blank()) + ylim(-1.5, 2.5) + xlim(-1.5,2.5) +
  geom_text(x=-1, y=2, label="1(b)", size=4) +
  geom_text(x=0, y=-1, label="Controlling for X identifies the effect of T,\ncontrolling for F generates post-treatment bias", size=3) +  geom_text(x=-1, y=-.5, label=expression(t[1]), size=3) +
  geom_text(x=0, y=-.5, label=expression(t[2]), size=3) +
  geom_text(x=1, y=-.5, label=expression(t[3]), size=3) +
  geom_text(x=2, y=-.5, label=expression(t[4]), size=3)



dag1c <- dagify(Y ~ T + F + X,
              F ~ T + U,
              T ~ U + X,
              coords = coords)
print( adjustmentSets( dag1c, "T", "Y" ) )

g1ctest <- ggdag(dag1c, node_size=8, text_size = 4) + 
  theme_classic() + remove_axes() + theme(axis.line=element_blank()) + ylim(-1.5, 2.5) + xlim(-1.5,2.5) +
  geom_text(x=-1, y=2, label="1(c)", size=4) +
  geom_text(x=0, y=-1, label="The effect of T is not identifiable\nusing observed covariates", size=3) +
  geom_text(x=-1, y=-.5, label=expression(t[1]), size=3) +
  geom_text(x=0, y=-.5, label=expression(t[2]), size=3) +
  geom_text(x=1, y=-.5, label=expression(t[3]), size=3) +
  geom_text(x=2, y=-.5, label=expression(t[4]), size=3)


figure1 <- arrangeGrob(g1atest, g1btest, g1ctest, ncol = 2, 
                       layout_matrix=rbind(c(1,1,2,2), c(NA, 3, 3, NA)))
figure1.ann <- grid.arrange(figure1, 
             sub=textGrob("HPT seek to estimate the effect of Distance to camps (T) on Intolerance (Y).\nThe issue is whether Länder fixed effects (F) create post-treatment bias, or whether they help to capture\nunobserved state-level factors (U) that also explain intolerance.",
                          x=unit(.1,"npc"), y=unit(.8,"npc"), gp = gpar(fontsize=9), just = "left"), heights = c(10, 1))
ggsave("figure1.pdf", figure1.ann,  height=7.5, width=7)





## FIGURE 2

coords <- list(
  x = c(Y=2, T=0, U1=-1, U2=0, F=1, X=0, T1=-1, T2=0),
  y = c(Y=1, T=1, U1=1.5, U2=2, F=1.5, X=0, T1=-1, T2=-1)
)

dag2a <- dagify(Y ~ T + F + X,
              F ~ U2,
              T ~ U1 + U2 + X,
              coords = coords)
print( adjustmentSets( dag2a, "T", "Y" ) )


g2atest <- ggdag(dag2a, node_size=8, text_size = 4) + 
  theme_classic() + remove_axes() + theme(axis.line=element_blank()) + ylim(-1.5, 2.5) + xlim(-1.5,2.5) +
  geom_text(x=-1, y=2, label="2(a)", size=4) +
  geom_text(x=0, y=-1, label="Controlling for F, X identifies the effect of T", size=3) +  geom_text(x=-1, y=-.5, label=expression(t[1]), size=3) +
  geom_text(x=0, y=-.5, label=expression(t[2]), size=3) +
  geom_text(x=1, y=-.5, label=expression(t[3]), size=3) +
  geom_text(x=2, y=-.5, label=expression(t[4]), size=3)



coords <- list(
  x = c(Y=2, T=0, U1=-1, U2=0, F=1, X=0, T1=-1, T2=0),
  y = c(Y=1, T=1, U1=1.5, U2=2, F=1.25, X=0, T1=-1, T2=-1)
)

dag2b <- dagify(Y ~ T + X + U2,
                F ~ U1 + U2,
                T ~ U1 + X,
                coords = coords)
print( adjustmentSets( dag2b, "T", "Y" ) )


g2btest <- ggdag(dag2b, node_size=8, text_size = 4) + 
  theme_classic() + remove_axes() + theme(axis.line=element_blank()) + ylim(-1.5, 2.5) + xlim(-1.5,2.5) +
  geom_text(x=-1, y=2, label="2(b)", size=4) +
  geom_text(x=0, y=-1, label="Controlling for X identifies the effect of T,\ncontrolling for F generates M-bias", size=3) +  geom_text(x=-1, y=-.5, label=expression(t[1]), size=3) +
  geom_text(x=0, y=-.5, label=expression(t[2]), size=3) +
  geom_text(x=1, y=-.5, label=expression(t[3]), size=3) +
  geom_text(x=2, y=-.5, label=expression(t[4]), size=3)



dag2c <- dagify(Y ~ T + F + X + U2 + F,
              F ~ U2 + U1,
              T ~ U1 + X,
              coords = coords)
print( adjustmentSets( dag2c, "T", "Y" ) )


g2ctest <- ggdag(dag2c, node_size=8, text_size = 4) + 
  theme_classic() + remove_axes() + theme(axis.line=element_blank()) + ylim(-1.5, 2.5) + xlim(-1.5,2.5) +
  geom_text(x=-1, y=2, label="2(c)", size=4) +
  geom_text(x=0, y=-1, label="The effect of T is not identifiable\nusing observed covariates", size=3) +
  geom_text(x=-1, y=-.5, label=expression(t[1]), size=3) +
  geom_text(x=0, y=-.5, label=expression(t[2]), size=3) +
  geom_text(x=1, y=-.5, label=expression(t[3]), size=3) +
  geom_text(x=2, y=-.5, label=expression(t[4]), size=3)




dag2d <- dagify(Y ~ T + X + U2,
              F ~ U2 + U1,
              T ~ U1 + X + U2,
              coords = coords)
print( adjustmentSets( dag2d, "T", "Y" ) )


g2dtest <- ggdag(dag2d, node_size=8, text_size = 4) + 
  theme_classic() + remove_axes() + theme(axis.line=element_blank()) + ylim(-1.5, 2.5) + xlim(-1.5,2.5) +
  geom_text(x=-1, y=2, label="2(d)", size=4) +
  geom_text(x=0, y=-1, label="The effect of T is not identifiable\nusing observed covariates", size=3) +
  geom_text(x=-1, y=-.5, label=expression(t[1]), size=3) +
  geom_text(x=0, y=-.5, label=expression(t[2]), size=3) +
  geom_text(x=1, y=-.5, label=expression(t[3]), size=3) +
  geom_text(x=2, y=-.5, label=expression(t[4]), size=3)



figure2 <- arrangeGrob(g2atest, g2btest, g2ctest, g2dtest, ncol = 2)
figure2.ann <- grid.arrange(figure2, 
                            sub=textGrob("HPT seek to estimate the effect of Distance to camps (T) on Intolerance (Y).\nThe issue is whether Länder fixed effects (F) create M-bias, or whether they help to capture\nunobserved state-level factors (U) that also explain intolerance.",
                                         x=unit(.1,"npc"), y=unit(.8,"npc"), gp = gpar(fontsize=9), just = "left"), heights = c(10, 1))

ggsave("figure2.pdf", figure2.ann,  height=7.5, width=7)
