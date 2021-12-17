let csvConst = "data:text/csv;charset=utf-8,";

d.rawData_.forEach(function (rowArray) {
    let dateVal = new Date(rowArray[0]);
    let tweetVal = rowArray[1];
    csvConst += dateVal.toISOString() + ',' + tweetVal + "\n";
});
let encodedUri = encodeURI(csvConst);
window.open(encodedUri);
