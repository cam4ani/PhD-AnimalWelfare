plotRel <- function (ZV, tit, miWe, dat= kuhMasse.df, sim= 1000) {
  library (gmodels)
  library (leiv)
  library (boot)
  library(lme4)
  library(pbkrtest)
  browser()
  # tryCatch({
  #### Funktionsparameter:
  ####
  #### ZV: Zielvariable (Körpermass)
  #### tit: Titel (Text für Körpermass)
  #### miWe: mittlerer Wert für Körpermass (definiert 0-Punkt)
  #### dat: Daten
  
  #############################################################################################
  ## Vorbereitung
  dat <- dat [dat [, 'Koerpermasse'] == ZV, ]
  dat [, 'NrMensch'] <- factor (as.character (dat [, 'NrMensch']))
  dat [, 'Kuhnummer'] <- factor (as.character (dat [, 'Kuhnummer']))
  
  kuehe <- levels (dat [, 'Kuhnummer'])
  kuehe.pred <- data.frame (Kuhnummer= kuehe)
  n.kuehe <- length (kuehe)
  obs <- levels (dat [, 'NrMensch'])
  obs.pred <- expand.grid (Kuhnummer= kuehe, NrMensch= obs)
  
  dat [, 'ZV.cent'] <- dat [, 'value'] - miWe
  # alternativ:
  # dat [, 'ZV.cent'] <- log (dat [, ZV] / miWe )
  
  library (lme4)
  library (pbkrtest)
  
  #############################################################################################
  ## Modell für Messband
  mb.lmer <- lmer (ZV.cent ~ (1 | Kuhnummer) + (1 | NrMensch), dat, subset= Method == 'tape')
  
  ## Residuen
  jpeg (paste ('Residuen/', ZV, '_messband.jpg', sep= ''), width= 480, height= 480)
  par (mfrow= c (2, 2), las= 1, bty= 'n')
  qqnorm (resid (mb.lmer))
  qqnorm (ranef (mb.lmer) [['Kuhnummer']] [, 1])
  qqnorm (ranef (mb.lmer) [['NrMensch']] [, 1])
  scatter.smooth (fitted (mb.lmer), resid (mb.lmer))
  dev.off ()
  
  ## Extraktion Varianzkomponenten
  vc.df <- data.frame (Mass= ZV,
                       method= 'tape',
                       error= c ('obs', 'mess'),
                       var= c (as.data.frame (VarCorr (mb.lmer)) [2, 4],
                               as.data.frame (VarCorr (mb.lmer)) [3, 4]))
  
  ## Schätzung für Kühe
  kuehe.df <- cbind (kuehe.pred,
                     data.frame (mb= predict (mb.lmer, obs.pred, re.form= ~ (1 | Kuhnummer))[1:n.kuehe]))
  
  ## Schätzung für observer / kühe
  obs.df <- cbind (obs.pred,
                   data.frame (mb= predict (mb.lmer, obs.pred, re.form= ~ (1 | Kuhnummer) + (1 | NrMensch))))
  
  
  #############################################################################################
  ## Modell für Matlab
  ml.lmer <- lmer (ZV.cent ~ gerade + (1 | Kuhnummer/Wiederholung) + (1 | NrMensch), dat, subset= Method == 'MATLAB')
  ml.red.lmer <- lmer (ZV.cent ~ (1 | Kuhnummer/Wiederholung) + (1 | NrMensch), dat, subset= Method == 'MATLAB')
  ## Daniel Hoop, hpda, 2018-05-25. Diesen Block ausgeklammert, weil es einen Fehler bei Argumenten ZV='IW', tit='Ischium width', miWe=30, gibt
  if(FALSE){
    ml.p <- PBmodcomp (ml.lmer, ml.red.lmer, nsim= sim)
    cat ('############################################################################################\n##', ZV, '\n')
    cat ('## P-Wert für "gerade stehen" (Matlab Auswertung):\n')
    print (summary (ml.p))
  }
  
  ## Residuen
  jpeg (paste ('Residuen/', ZV, '_matlab.jpg', sep= ''), width= 720, height= 480)
  par (mfrow= c (2, 3), las= 1, bty= 'n')
  qqnorm (resid (ml.lmer))
  qqnorm (ranef (ml.lmer) [['Kuhnummer']] [, 1])
  qqnorm (ranef (ml.lmer) [['Wiederholung:Kuhnummer']] [, 1])
  qqnorm (ranef (ml.lmer) [['NrMensch']] [, 1])
  scatter.smooth (fitted (ml.lmer), resid (ml.lmer))
  boxplot (split (resid (ml.lmer), dat [dat [, 'Method'] == 'MATLAB', 'gerade']))
  dev.off ()
  
  ## Extraktion Varianzkomponenten
  vc.df <- rbind (vc.df,
                  data.frame (Mass= ZV,
                              method= 'matlab',
                              error= c ('obs', 'mess'),
                              var= c (as.data.frame (VarCorr (ml.lmer)) [3, 4],
                                      as.data.frame (VarCorr (ml.lmer)) [4, 4])))
  
  ## Schätzung für Kühe
  kuehe.df <- cbind (kuehe.df,
                     data.frame (ml= predict (ml.lmer, cbind (obs.pred, data.frame (gerade= 0.5, Wiederholung= '1')), re.form= ~ (1 | Kuhnummer))[1:n.kuehe]))
  
  ## Schätzung für observer / kühe
  obs.df <- cbind (obs.df,
                   data.frame (ml= predict (ml.lmer, cbind (obs.pred, data.frame (gerade= 0.5, Wiederholung= '1')), re.form= ~ (1 | Kuhnummer) + (1 | NrMensch))))
  
  detach ('package:pbkrtest')
  # detach ('package:lmerTest')
  detach ('package:lme4')
  
  
  #############################################################################################
  ## Graphik
  
  ## Graphik Aufsetzen
  mimax <- range (as.matrix (kuehe.df [, 2:3]))
  mimax <- mimax - c (-1, 1) * 0.3 * (mimax [1])
  scatter.smooth (kuehe.df [, 'mb'], kuehe.df [, 'ml'], pch= 16, xlim= mimax, ylim= mimax,
                  xlab= paste (miWe, 'cm'), ylab= tit, lwd= 2, lpars= list (lty= '11', lwd= 2))
  abline (0, 1, lty= '34')
  
  ## Einzelmessungen
  for (k in kuehe){
    points (rep (kuehe.df [kuehe.df [, 'Kuhnummer'] == k, 'mb'], length (dat [dat [, 'Kuhnummer'] == k & dat [, 'Method'] == 'MATLAB', 'ZV.cent'])),
            dat [dat [, 'Kuhnummer'] == k & dat [, 'Method'] == 'MATLAB', 'ZV.cent'], pch= '.')
    
    points (dat [dat [, 'Kuhnummer'] == k & dat [, 'Method'] == 'tape', 'ZV.cent'],
            rep (kuehe.df [kuehe.df [, 'Kuhnummer'] == k, 'ml'], length (dat [dat [, 'Kuhnummer'] == k & dat [, 'Method'] == 'tape', 'ZV.cent'])), pch= '.')
  }
  
  ## Beobachter
  for (k in kuehe){
    points (rep (kuehe.df [kuehe.df [, 'Kuhnummer'] == k, 'mb'], length (obs.df [obs.df [, 'Kuhnummer'] == k, 'ml'])),
            obs.df [obs.df [, 'Kuhnummer'] == k, 'ml'])
    
    points (obs.df [obs.df [, 'Kuhnummer'] == k, 'mb'],
            rep (kuehe.df [kuehe.df [, 'Kuhnummer'] == k, 'ml'], length (obs.df [obs.df [, 'Kuhnummer'] == k, 'mb'])))
  }
  
  
  library (nlme)
  #############################################################################################
  ## Schätzung, CI Intercept & Steigung, p-Wert Intercept
  rel.gls <- gls (ml ~ mb, kuehe.df)
  cat ('## Modell Matlab gegen Tape:\n')
  cat ('## p-Wert Intercept und Schätzung Intercept / Steigung\n')
  print (intervals (rel.gls))
  print (anova (rel.gls))
  
  ## Residuen
  jpeg (paste ('Residuen/', ZV, '_rel.jpg', sep= ''), width= 480, height= 240)
  par (mfrow= c (1, 2), las= 1, bty= 'n')
  qqnorm (resid (rel.gls))
  scatter.smooth (fitted (rel.gls), resid (rel.gls))
  dev.off ()
  
  ## Graphik: Schätzkurve (inkl. CI?)
  abline (rel.gls, lwd= 3)
  rel.pred <- data.frame (1, seq (mimax [1], mimax [2], len= 100))
  names (rel.pred) <- names (coef (rel.gls))
  rel.pred <- cbind (rel.pred,
                     gmodels::estimable (lm (ml ~ mb, kuehe.df), rel.pred, conf.int= 0.95))
  
  lines (Lower.CI ~ mb, rel.pred, lwd= 2)
  lines (Upper.CI ~ mb, rel.pred, lwd= 2)
  
  ## p-Wert Steigung == 1
  rel.s1.gls <- gls ((ml-mb) ~ mb, kuehe.df)
  cat ('## Modell Matlab gegen Tape:\n')
  cat ('## p-Wert für Steigung =? 1, p-Wert für mb\n')
  print (anova (rel.s1.gls))
  
  ## Residuen
  jpeg (paste ('Residuen/', ZV, '_rel_s1.jpg', sep= ''), width= 480, height= 240)
  par (mfrow= c (1, 2), las= 1, bty= 'n')
  qqnorm (resid (rel.s1.gls))
  scatter.smooth (fitted (rel.s1.gls), resid (rel.s1.gls))
  dev.off ()
  
  detach ('package:nlme')
  
  #############################################################################################
  ## Ausgabe Varianzkomponenten
  # },
  # error=function(e) browser() )
  if (exists("vc.df"))
    return (invisible(vc.df))
  return (NA)
}
