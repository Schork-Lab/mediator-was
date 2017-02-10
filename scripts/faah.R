library(dplyr)
library(ggbio)
library(ggplot2)
library(reshape2)
library(Cairo)

library(frea)

bayesian <- (read.delim('/home/unix/aksarkar/incoming/FAAH/FAAH.bayesian_prior_and_posterior.tsv') %>%
             dplyr::group_by(position) %>%
             dplyr::do(data.frame(position=rep(.$position, length.out=2), variable=c('Prior', 'Posterior'), y=rbind(.$prior_mean, .$prior_sd), sd=rbind(.$prior_sd, .$sd))) %>%
             dplyr::select(position, variable, y=y, sd=sd) %>%
             dplyr::mutate(ymin=y - sd, ymax=y + sd, method='bayesian'))
enet <- (read.delim('/home/unix/aksarkar/incoming/FAAH/FAAH.elasticnet.tsv') %>%
         dplyr::select(position, variable=bootstrap, value=beta) %>%
         dplyr::filter(variable == 'full') %>%
         dplyr::select(position, variable, y=value)
models <- rbind(bayesian, enet)

locus_plot <- (ggplot(bayesian, aes(x=position, y=y, color=factor(variable))) +
               labs(y='Estimated eQTL effect size', x='Position on chromosome 1',
                    color='Model') +
               geom_vline(xintercept=unique(models$position), color='gray90', size=I(.25)) +
               geom_pointrange(aes(ymin=ymin, ymax=ymax), size=I(.25), fatten=.5, position='jitter') +
               geom_point(data=enet) +
               geom_hline(yintercept=0, size=I(.25)) +
               scale_color_brewer(palette='Dark2') +
               theme_nature +
               theme(panel.margin=unit(2, 'mm'),
                     legend.position='right'))
Cairo(file='/broad/compbio/aksarkar/projects/mediator-was/faah-models.pdf', type='pdf', width=120, height=89, units='mm')
print(locus_plot)
dev.off()

liab <- (read.delim('/broad/hptmp/aksarkar/nwas/FAAH/FAAH.liab.tsv') %>%
         dplyr::mutate(case=phen >= 3.09023230617))
pred_expr <- (read.delim('/home/unix/aksarkar/incoming/FAAH/FAAH.predicted_expression.tsv') %>%
              dplyr::select(X, prior, posterior) %>%
              reshape2::melt(c('X')) %>%
              dplyr::inner_join(., liab, by='X') %>%
              dplyr::group_by(variable, case) %>%
              dplyr::do(data.frame(variable=.$variable, case=.$case, x=.$value[order(.$value)], y=seq(1 / nrow(.), 1, 1/nrow(.)))))
levels(pred_expr$variable) <- c('Prior', 'Posterior')

cdf_plot <- (ggplot(pred_expr, aes(x, y, color=case)) +
        labs(x='Predicted expression', y='Cumulative fraction', color='CD') +
        geom_line() +
        facet_grid(variable ~ .) +
        theme_nature +
        theme(panel.margin=unit(2, 'mm'),
              legend.position='right'))
Cairo(file='/broad/compbio/aksarkar/projects/mediator-was/faah-expr.pdf', type='pdf', width=89, height=40, units='mm')
print(cdf_plot)
dev.off()
