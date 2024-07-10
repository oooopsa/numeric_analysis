document.addEventListener('DOMContentLoaded', function() {
    // 获取按钮元素
    var jumpButton = document.getElementById('jumpButton');
    
    // 为按钮添加点击事件监听器
    jumpButton.addEventListener('click', function() {
        // 设置要跳转的页面URL
        window.location.href = "NumericAnalysisApi.html";
    });
});