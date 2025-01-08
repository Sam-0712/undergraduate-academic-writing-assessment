document.getElementById('testButton').addEventListener('click', function() {
    document.getElementById('title').value = '测试标题';
    document.getElementById('abstract').value = '这是一个测试摘要。';
    document.getElementById('keywords').value = '测试, 示例, 代码';
    document.getElementById('content').value = '这是测试正文内容。';
    document.getElementById('date').value = '2023-10-01'; // 示例日期
});
