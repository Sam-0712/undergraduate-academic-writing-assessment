document.getElementById('scoreForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const title = document.getElementById('title').value;
    const abstract = document.getElementById('abstract').value;
    const keywords = document.getElementById('keywords').value;
    const content = document.getElementById('content').value;

    // 检查以上几项是否为空，只要有一项为空，则提示用户重新输入
    if (!title || !abstract || !keywords || !content) {
        alert('请填写所有必填项！');
        return;
    }

    // 获取日期输入
    const dateInput = document.getElementById('date').value;

    // 检查日期是否为空
    if (!dateInput) {
        alert('请选择日期！');
        return;
    }

    // 解析日期
    const dateParts = dateInput.split('-');
    const year = dateParts[0];
    const month = dateParts[1] || '01'; // 如果月份缺失，默认为01
    const day = dateParts[2] || '01';   // 如果日期缺失，默认为01

    // 检查年份是否为空
    if (!year) {
        alert('年份不能为空，请重新输入日期！');
        return;
    }

    // 构造完整的日期字符串
    const fullDate = `${year}-${month}-${day}`;
    const dateInputObj = new Date(fullDate);
    const startDate = new Date('2022-09-01');

    // 计算天数差
    const dayDiff = Math.min(1400, Math.ceil((dateInputObj - startDate) / (1000 * 60 * 60 * 24)));

    // 计算字数
    const wc = content.length;

    // 计算分数
    let score;
    const score_p = 94.348 - 0.061 * dayDiff + 0.0000834 * dayDiff * dayDiff + 0.001 * wc;
    if (score_p < 92) {
        score = score_p;
    } else {
        score = 92 + Math.min(5, Math.log(Math.max(1, (score_p - 92))));
    }

    // 显示结果
    document.getElementById('result').innerText = `预估分数：${score.toFixed(1)}`;
});
