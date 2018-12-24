require 'open-uri'
require 'json'
require 'daru'
require 'distribution'
require 'sqlite3'
require 'test/unit/assertions'

include Test::Unit::Assertions

def roc_curve(scores)
  ### BEGIN SOLUTION
  total_neg = scores.inject(0.0) {|u,s| u += (1 - s.last)}
  total_pos = scores.inject(0.0) {|u,s| u += s.last}
  c_neg = 0.0
  c_pos = 0.0
  fp = [0.0]
  tp = [0.0]
  auc = 0.0
  scores.sort_by {|s| -s.first}.each do |s|
    c_neg += 1 if s.last <= 0
    c_pos += 1 if s.last > 0  

    fpr = c_neg / total_neg
    tpr = c_pos / total_pos
    auc += 0.5 * (tpr + tp.last) * (fpr - fp.last)
    fp << fpr
    tp << tpr
  end
  ### END SOLUTION
  return [fp, tp, auc]
end
