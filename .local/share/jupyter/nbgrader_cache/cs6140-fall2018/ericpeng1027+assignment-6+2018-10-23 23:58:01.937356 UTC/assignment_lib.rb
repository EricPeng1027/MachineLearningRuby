require 'test/unit/assertions'
require 'daru'
require 'distribution'
require 'json'

include Test::Unit::Assertions

## Loads data files
def read_sparse_data_from_csv prefix
  data = []
  classes = Hash.new {|h,k| h[k] = 0}
  header = File.read(prefix + ".header").chomp.split(",")  
  
  File.open(prefix + ".csv").each_line.with_index do |l, i|
    a = l.chomp.split ","
    next if a.empty?
    row = {"features" => Hash.new}
    
    header.each.with_index do |k, i|
      v = a[i].to_f
      if k == "label"
        row["label"] = v.to_f
      else
        next if v.zero?
        row["features"][k] = v
      end
    end

    row["features"]["1"] = 1
    data << row
  end
  return {"classes" => classes, "features" => header[0,header.size - 1] + ["1"], "data" => data}
end

def cross_validate data, folds, &block
  dataset = data["data"]
  fold_size = dataset.size / folds
  subsets = []
  dataset.shuffle.each_slice(fold_size) do |subset|
    subsets << subset
  end
  i_folds = Array.new(folds) {|i| i}
  
  i_folds.collect do |fold|
    test = subsets[fold]
    train = (i_folds - [fold]).flat_map {|t_fold| subsets[t_fold]}
    train_data = data.clone
    train_data["data"] = train
    
    test_data = data.clone
    test_data["data"] = test
    
    yield train_data, test_data, fold
  end
end

def mean x
  sum = x.inject(0.0) {|u,v| u += v}
  sum / x.size
end

def stdev x
  m = mean x
  sum = x.inject(0.0) {|u,v| u += (v - m) ** 2.0}
  Math.sqrt(sum / (x.size - 1))
end
