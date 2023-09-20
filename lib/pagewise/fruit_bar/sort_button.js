// js を init 時点で走らせるのはBokeh 3.1 以降 しかできないようなので、source_default, source のように2つ用意して選択されていないときはsource_defaultを使うようにする
// js を init で走らせるHookはこのPRで対応された：https://github.com/bokeh/bokeh/pull/12370
function sortWordsByFreq(words, frequencies) {
    // freqを降順にしてwordsも同じ順番にする
    let sorted = [];
    for (let i = 0; i < frequencies.length; i++) {
        sorted.push([frequencies[i], words[i]]);
    }
    sorted.sort(function(a, b) {
        return b[0] - a[0];
    });
    frequencies = [];
    words = [];
    for (let i = 0; i < sorted.length; i++) {
        frequencies.push(sorted[i][0]);
        words.push(sorted[i][1]);
    }
    return [words, frequencies];
}

function make_dict(words, frequencies) {
    // words, frequencies の pair から 連想配列 を作る
    let dict = {};
    for (let i = 0; i < words.length; i++) {
        if (frequencies[i] !== undefined)
            dict[words[i]] = frequencies[i];
    }
    return dict;
}
let data = source.data;
let figure = plot;
let words = data['words'];
let freqSelected = data['選択した文書の単語頻度'];
let freqAll = data['全体の単語頻度'];
// debugger;
let dictFreqSelected = make_dict(words, freqSelected);
let dictFreqAll = make_dict(words, freqAll);
let newWords, newFreqSelected;
[newWords, newFreqSelected] = sortWordsByFreq(words, freqSelected);
// debugger;
let newFreqAll = newWords.map(function(w) {
    return dictFreqAll[w];
});
source.data['words'] = newWords
source.data['全体の単語頻度'] = newFreqAll;
source.data['選択した文書の単語頻度'] = newFreqSelected;

source.change.emit();
plot.y_range.factors = source.data['words'];
// plot.change.emit();
