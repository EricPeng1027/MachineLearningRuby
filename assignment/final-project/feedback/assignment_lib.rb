require 'open-uri'
require 'json'
require 'daru'
require 'distribution'
require 'sqlite3'
require 'test/unit/assertions'

include Test::Unit::Assertions

def roc_curve(scores)
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
  return [fp, tp, auc]
end

def get_labels_for db, predictions
  ids = predictions.keys.join(", ")
  sql = "select sk_id_curr, target from application_train where sk_id_curr in (#{ids})"
  scores = Array.new
  db.execute(sql) do |row|
    y_hat = predictions[row["SK_ID_CURR"]]
    y = row["TARGET"]
    scores << [y_hat, y]
  end
  return scores
end

def plot_roc_curve fp, tp, auc
  plot = Daru::DataFrame.new({x: fp, y: tp}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "False Positive Rate"
    plot.y_label "True Positive Rate"
    diagram.title("AUC: %.4f" % auc)
    plot.legend(true)
  end
end  
  
