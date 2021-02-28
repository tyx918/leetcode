# 合并两个有序链表
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
示例：

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* ansnode = new ListNode(-1);//
        ListNode* pre = ansnode;
        while(l1 && l2) {
            if(l1->val <= l2->val) {
                pre->next = l1;
                l1 = l1->next;
            } else {
                pre->next = l2;
                l2 = l2->next;
            }
            pre = pre->next;
        }
        pre->next = (l1!=nullptr)?l1:l2;
        return ansnode->next;
    }
};
```

```js
/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
var mergeTwoLists = function(l1, l2) {
    let ans = new ListNode();
    let cur = ans;
    while(l1 && l2) {
        if(l1.val <= l2.val) {
            cur.nex+-t = l1;
            l1 = l1.next;
        } else {
            cur.next = l2;
            l2 = l2.next;
        }
        cur = cur.next;
    }
    cur.next = l1?l1:l2;
    return ans.next;
};
```

总结：关于链表的问题,可以先声明一个头结点head，再声明一个cur的当前指针指向它，这个结点用来移动。(注意返回的是head.next)
以上是迭代法，由于要参加面试，故现只记住迭代。

# 两数之和
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int car = 0;
        ListNode* ans = new ListNode(0);//答案结点
        ListNode* cur = ans;//当前结点
        while(l1 || l2) {
            int m  = l1?l1->val:0;
            int n  = l2?l2->val:0;
            int sum = car + m + n;
            cur->next = new ListNode(sum%10);
            car = sum/10;
            cur = cur->next;
            if(l1) l1 = l1->next;
            if(l2) l2 = l2->next;
        }
        if(car > 0) {
            cur->next = new ListNode(car);
        }
        return ans->next;
    }
};
```

# 两数之和
算法：利用哈希表的键值对可完成快速的访问和查找。数组的值：键，数组的下标是：值
```cpp
/*
*unordered_map无序的，利用哈希表来实现，搜索时间为log(n)
*map是在缺省的情况下，是有序的，利用二红黑树（一种平衡搜索树）,搜索时间为：O(1)为平均时间，最坏情况下的时间复杂度为O（n）
*/
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int len = nums.size();
        unordered_map<int,int> myMap;
        for(int i = 0; i < len; i++) {
            auto it = myMap.find(target-nums[i]);
            //如果不存在则返回myMap.end()
            if(it!=myMap.end()) {
                return {it->second , i};
            }
            //如果存在，则返回当前的迭代器。first是指key ， second是指的是value
            myMap.insert({nums[i] , i});
        }
        return {};
    }
};
```

# 两个栈实现队列
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead，如为空则返回-1.

算法：一个栈出，一个栈进，将栈进的元素推入栈出元素，即可实现对列的先进先出。需要注意的是，必须当出的栈为空的时候，才把进栈中的元素都推入出栈的元素。
```cpp
class CQueue {
public:
    //声明两个栈
    stack<int> stack_in , stack_out; 
    CQueue() {
        // 清空两个栈
        while(!stack_in.empty()) {
            stack_in.pop();
        }
        while(!stack_out.empty()) {
            stack_out.pop();
        }
    }
    
    void appendTail(int value) {
        //推入值
        stack_in.push(value);
    }
    
    int deleteHead() {
        //当出栈的清空的时候才一次性将入栈的元素推入，如果不清空的话，后面的元素会覆盖上去
        if(stack_out.empty()) {
            while(!stack_in.empty()) {
                stack_out.push(stack_in.top());
                stack_in.pop();
            }
        }
        if(stack_out.empty()) {
            return -1;
        }else {
            int deletItem = stack_out.top();//获取顶端元素
            stack_out.pop();//弹出顶端元素
            return deletItem;
        }
    }
};
```


# 无重复字符的最长子串
滑动窗口：思路和算法：
https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/solution/wu-zhong-fu-zi-fu-de-zui-chang-zi-chuan-by-leetc-2/
双指针：左指针固定，右指针往右移动。while循环中可以固定长度，当不满足条件的时候跳出即可！
```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.empty()) return 0;
        int len = s.length();
        int ans = 0 , rk = 0;//rk表示右指针的位置
        unordered_set<char> occ;
        //滑动窗口：左指针固定，右指针递增。
        for(int i = 0;i < len;i++) {
            if(i != 0) {
                occ.erase(s[i - 1]);
            }
            while(rk < len && !occ.count(s[rk])) {
                 occ.insert(s[rk]);
                 rk++;
            }
            ans = max(ans , rk - i);
        }
        return ans;
    }
};
```
```js
/**
 * @param {string} s
 * @return {number}
 */
var lengthOfLongestSubstring = function(s) {
    const occ = new Set();
    let len = s.length,ans = 0;   
    let rk = 0;//右指针
    //i表示左指针
    for(let i = 0;i < len;i++) {
        if(i !== 0) {
            occ.delete(s[i - 1])
        }
        while(rk < len && !occ.has(s[rk])) {
            occ.add(s[rk]);
            rk++;
        }
        //occ的长度即为子串长度
        ans = Math.max(ans,occ.size);
    }
    return ans;
};
```

# 53 最长子序和
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
动态规划：sum[i] = max(sum[i-1],num[i])
```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        int max_ending = nums[0];
        int max_sofar = nums[0];
        for(int i = 1;i < len;i++) {
            max_ending = max(max_ending + nums[i] , nums[i]);
            max_sofar = max(max_ending , max_sofar);
        }
        return max_sofar;
    }
};
```

# 70 爬楼梯
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
算法：动态规划的经典算法。状态转移方程：f(x) = f(x-1) + f(x-2).
```cpp
class Solution {
public:
    int climbStairs(int n) {
        if( n == 1) return 1;
        if( n == 2) return 2;
        vector<int> dp;
        dp.reserve(n);
        dp.push_back(1);
        dp.push_back(1);
        for(int i = 2;i <= n;i++) {
            dp.push_back(dp[i -1] + dp[i -2]);//根据状态方程来写
        }
        return dp[n];
    }
};
```
```js
/**
 * @param {number} n
 * @return {number}
 */
var climbStairs = function(n) {
    let dp = [];
    dp[0] = 1;
    dp[1] = 1;
    if(n===1) return dp[1];
    for(let i = 2;i <= n;i++) {
        dp[i] = dp[i -1] + dp[i -2];
    }
    return dp[n];
};
```

# 101 对称二叉树
算法：二叉树的先序遍历
```cpp

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
递归法
class Solution {
public:

    bool help(TreeNode* rootl ,TreeNode* rootr) {
        if(rootl == NULL && rootr == NULL) return true;
        if(rootl == NULL || rootr == NULL) return false;
        if(rootl->val != rootr->val) return false;
       return  help(rootl->left , rootr->right)&&help(rootl->right , rootr->left); 
    }

    bool isSymmetric(TreeNode* root) {
       return help(root, root);
    }
};

迭代法

```

# 树的遍历(dfs , bfs)

### 前序 以https://leetcode-cn.com/problems/binary-tree-preorder-traversal/为例
```cpp (递归)
    void help(TreeNode* node , vector<int>& ans) {
        if(node == nullptr) return;
        ans.push_back(node->val);
        help(node->left , ans);
        help(node->right , ans);
    }
```

``` cpp 迭代
    vector<int> preorderTraversal(TreeNode* root) {
        // //迭代
        // vector<int> ans;
        // help(root , ans);
        // return ans;
        vector<int> ans;
        if(root == nullptr) return ans;
        stack<TreeNode*> myStack;
        myStack.push(root);
        while(!myStack.empty()) {
            TreeNode* temp = myStack.top();
            myStack.pop();
            ans.push_back(temp->val);
            if(temp->right) myStack.push(temp->right);
            if(temp->left) myStack.push(temp->left);
        }
        return ans;
    }
```

```js (递归版)
var preorderTraversal = function(root) {
    let ans = [];
    const help = node => {
        if(!node) return;
        ans.push(node.val);
        help(node.left);
        help(node.right);
    }
    help(root);
    return ans;
};
```

```js (迭代版)
    let ans = [];
    if(root === null) return ans;
    let stack = [root];
    //数组的判断是不可以通过其（stack === null）实现
    while(stack.length !== 0) {
        const node = stack.pop();
        ans.push(node.val);
        if(node.right) stack.push(node.right);
        if(node.left) stack.push(node.left); 
    }
    return ans;
```

### 中序 以 https://leetcode-cn.com/problems/binary-tree-inorder-traversal/submissions/ 为例
```cpp (递归)
void help(TreeNode* root , vector<int>& ans) {
    if(root == nullptr) return;
    help(root->left , ans);
    ans.push_back(root->val);
    help(root->right , ans);
}
```
```cpp (迭代)
        vector<int> res;
        stack<TreeNode*> stk;
        while (root != nullptr || !stk.empty()) {
            while (root != nullptr) {
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            res.push_back(root->val);
            root = root->right;
        }
        return res;
```

### 后序遍历 以https://leetcode-cn.com/problems/binary-tree-postorder-traversal/submissions/为例
```cpp (递归)
    void help(TreeNode* root , vector<int>& ans) {
        if(root == nullptr) return;
        help(root->left , ans);
        help(root->right , ans);
        ans.push_back(root->val);
    }
```

```cpp (迭代)
    vector<int> res;
    if (root == nullptr) {
        return res;
    }
    stack<TreeNode*> stk;
    TreeNode *prev = nullptr;
    while (root != nullptr || !stk.empty()) {
        while (root != nullptr) {
            stk.emplace(root);
            root = root->left;
        }
        root = stk.top();
        stk.pop();
        if (root->right == nullptr || root->right == prev) {
            res.emplace_back(root->val);
            prev = root;
            root = nullptr;
        } else {
            stk.emplace(root);
            root = root->right;
        }
    }
    return res;
```

# 二叉树的最大深度
算法：利用层序遍历来计算深度，利用每一次的queue的长度来实现，每一层的遍历，一层遍历完结果+1即可

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:

    int maxDepth(TreeNode* root) {
        TreeNode* temp; //保存节点
        if(root == nullptr) return 0;
        queue<TreeNode*> myQueue;
        myQueue.push(root);
        int ans = 0;
        //获取当前子节点的所有数目
        while(!myQueue.empty()) {
            int size = myQueue.size();
            for(int i = 0;i < size;i++) {
                temp = myQueue.front();
                myQueue.pop();

                if(temp->left) myQueue.push(temp->left);
                if(temp->right) myQueue.push(temp->right);
            }
            ans++;
        }
        return ans;
    }
};
```

```js
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number}
 */
var maxDepth = function(root) {
    if(root == null) return 0;
    let temp = null , queue = [root];
    let ans = 0;
    //获取当前子节点的所有数目
    while(queue.length !== 0) {
        let len = queue.length;
        for(let i = 0;i < len;i++) {
            temp = queue.shift();
            if(temp.left) queue.push(temp.left);
            if(temp.right) queue.push(temp.right);
        }
        ans++;
    }
    return ans;
};
```
# 只出现一次的数字
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

示例 1:
输入: [2,2,1]
输出: 1

示例 2:
输入: [4,1,2,1,2]
输出: 4

算法：利用哈希表，出现一次则为false，出现两次则为true。第二次遍历找出来，值为false的那个。
```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        unordered_map<int , int> my_map;
        int ans = 0;
        for(int i = 0;i < nums.size();i++) {
            if(my_map.find(nums[i]) != my_map.end()) {
                my_map[nums[i]] = true;
            }
            else {
                my_map[nums[i]] = false;
            }
        }
        for(int j = 0;j< nums.size();j++) {
            if(my_map[nums[j]] == false) ans = nums[j];
        }
        return ans;
    }
};
```


# 第一个只出现一次的字符
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
示例:
s = "abaccdeff"
返回 "b"
s = "" 
返回 " "

算法：利用哈希表，出现一次的字符为true ，出现过多次的字符为false。
     第二次遍历的时候找出其第一个为true的字符。

```cpp
class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<int , bool> my_map;
        int len = s.length();
        for(int i = 0;i < len;i++) {
            if(my_map.find(s[i]) == my_map.end()) {
                my_map[s[i]] = true;
            } else {
                my_map[s[i]] = false;
            }
        }
        for(int j = 0;j < len;j++) {
            if(my_map[s[j]] == true)  return s[j];
        }
        return ' ';        
    }
};
```
```js
/**
 * @param {string} s
 * @return {character}
 */
var firstUniqChar = function(s) {
    let map = {};
    let len = s.length;
    console
    for(let i = 0;i < len;i++) {
        if(!map.hasOwnProperty(s[i])) {
            map[s[i]] = true;
        } else{
            map[s[i]] = false;
        }
    }
    for(let i = 0;i < len;i++) {
        if(map[s[i]] === true) return s[i];
    }
    return ' ';
};
```

# 反转链表
示例:

输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL

算法：一个pre ，一个temp，一个cur 。cur表示指向当前，pre表示指向前一个，temp用来保存cur的next节点。
```cpp (迭代版)
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        // ListNode* head = new ListNode(-1);
        ListNode* cur = head;
        ListNode* pre = NULL;//前一个需为空
        ListNode* temp;
        while(cur != NULL) {
            temp = cur->next;//保存下一个节点
            cur->next = pre;//当前的指向前一个

            pre = cur;//pre往前移
            cur = temp;//往前移，注意此时的cue的next已经指向了之前(pre)
        }
        return pre;
    }
};
```

# 环形链表
算法： 环形链表的判别：快慢指针，一个指针移动一次，另外一个指针移动两次，若有环，则必定相遇。
具体实现如下：一些注意的地方见注释
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(head == nullptr || head->next == nullptr) return false;
        ListNode* low = head;
        ListNode* fast = head.next; //为了进入while循环中，才让其指向next
        while(low != fast) {
            if(fast == nullptr || fast->next == nullptr) return false;//无环，判断fast即可，因为            //fast走得快，null的next是错误的，所以判断fast->next是否为空
            low = low->next;
            fast = fast->next->next;
        }
        //有环则相遇
        return true;
    }
};
```

# 最小栈
算法：利用数组实现栈的解构。
```cpp
class MinStack {
public:
    /** initialize your data structure here. */
    int MAX = 100;
    int count = 0;
    int *arr;
public:
    MinStack() {
        arr = new int[MAX];
    }
    
    void push(int x) {
        // if(co    unt >= 100) {
        //     MAX = 2*MAX;
        //     arr = new int[2*MAX];//扩容之后原来的不知道会不会变。
        //     for()
        // }
        arr[count] = x;
        this->count++;
    }
    
    void pop() {
        count--;
    }
    
    int top() {
        return arr[count-1];
    }
    
    int getMin() {
        int min = arr[0];
        for(int i = 0;i < count;i++) {
            if(arr[i] < min) {
                min = arr[i];
            }
        }
        return min;
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```

# 汉明距离
算法：整形可以直接像二进制一样操作，直接异或和相与以及右移。
```cpp
class Solution {
public:
    int hammingDistance(int x, int y) {
        int res = x^y;//整形可以直接异或
        int ans = 0;
        while(res != 0) {
            if(res&1) { //看最左边是不是为1
                ans++;
            }
            res = res >> 1;//往右移一位
        }
        return ans;
    }
};
```

# 相交链表
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

算法：先遍历节点将一个链表推入哈希表，然后再在哈希表中遍历另外一个链表是否存在即可。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        unordered_map<ListNode* , int> myMap;
        for(ListNode* curA = headA;curA != nullptr;curA = curA->next) {
            if(myMap.find(curA) == myMap.end()) {
                myMap[curA] = 1;
            }
        }
        for(ListNode* curB = headB;curB != nullptr;curB = curB->next) {
            if(myMap.find(curB)!=myMap.end()) {
                return curB;
            }
        }
        return nullptr;
    }
};
```

# 多数元素
给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。

示例 1: 输入: [3,2,3] 输出: 3
示例 2:输入: [2,2,1,1,1,2,2] 输出: 2

算法：利用哈希表存入数据，再遍历即可,时间复杂度o(2n),空间复杂度o(n)。
```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int len = nums.size();
        int yuzhi = len / 2 + 1;//阈值
        unordered_map<int , int> myMap;
        int ans = 0;
        for(int i = 0;i <len;i++) {
            if(myMap.find(nums[i]) == myMap.end()) {
                myMap[nums[i]] = 1;//不存在就为1
            }
            else{
                myMap[nums[i]]++;//存在就+1
            }
        }
        for(int i = 0;i < len;i++) {//寻找其符合的值
            if(myMap[nums[i]] >= yuzhi) ans = nums[i];
        }  
        return ans;
    }
};
```

# 198 打家劫舍
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

示例 1：
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。

算法：动态规划+滚动数组。转态方程如下：dp[i] = max(dp[i-2] + nums[i] , dp[i-1])
边界条件如下：dp[0] = nums[0] , dp[1] = max[nums[0] , nums[1]].

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        if(len == 1) return nums[0];//边界条件
        if(len == 2) return max(nums[0],nums[1]);//边界条件
        int* dp = new int[len];
        dp[0] = nums[0];
        dp[1] = max(nums[0],nums[1]);
        for(int i = 2;i < len;i++) {
            dp[i] = max(dp[i - 2] + nums[i] , dp[i-1]); //状态方程
        }
        return dp[len - 1];
    }
};
```

# 翻转二叉树
输入：
     4
   /   \
  2     7
 / \   / \
1   3 6   9
输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        // /*递归版本*/
        // if(root == nullptr) return nullptr;
        // //子节点进行交换
        // TreeNode* temp = root->left;
        // root->left = root->right;
        // root->right = temp;
        ////前序遍历
        // invertTree(root->left);
        // invertTree(root->right);
        
        // return root;
        
        if(root == nullptr) return nullptr;
        queue<TreeNode*> myQueue;
        myQueue.push(root);
        while(!myQueue.empty()) {
                //层序遍历
                TreeNode* tmp = myQueue.front();
                TreeNode* left = tmp->left;
                tmp->left = tmp->right;
                tmp->right = left;
                myQueue.pop();
                if(tmp->left) myQueue.push(tmp->left);
                if(tmp->right) myQueue.push(tmp->right);
        } 
        return root;
    }
};
```

# 合并二叉树

给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

算法：二叉树递归，注意结束条件。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        if (t1 == nullptr) {
            return t2;
        }
        if (t2 == nullptr) {
            return t1;
        }
        auto merged = new TreeNode(t1->val + t2->val);
        merged->left = mergeTrees(t1->left, t2->left);
        merged->right = mergeTrees(t1->right, t2->right);
        return merged; 
    }
};
```

# 回文链表
示例 1:

输入: 1->2
输出: false
示例 2:

输入: 1->2->2->1
输出: true

算法： 快慢指针。一快一慢。当一个指针指向尾的时候，另一个指针在中间。在slow指针移动的过程中反转链表即可。注意：
当为奇数的时候则fast指针不会为空，slow指针需要移动到下一个。当为偶数的时候，fast指针为空，slow指针在当前的位置。 时间复杂度：o(n) , 空间复杂度：o(1).

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return true;
        ListNode* slow = head;
        ListNode* fast = head;
        ListNode* pre = nullptr;
        ListNode* tmp;

        while(fast != nullptr && fast->next != nullptr) {
            fast = fast->next->next;//在反转链表之前写，免得指回去了
            //slow指针移动，同时反转链表
            tmp = slow->next;//
            slow->next = pre;//

            pre = slow;
            slow = tmp;
        }
        //当长度为奇数的时候，fast最后不会指向空，此时slow要往后移一位
        if(fast!=nullptr) {
            slow = slow->next;
        }
        //比较
        while(pre != nullptr && slow != nullptr) {
            if(pre->val != slow->val) return false;
            pre = pre->next;
            slow = slow->next;
        }
        return true;
    }
};
```

# 移动零(重点)
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

算法：双指针，从非零的数字出发，当right的值为非0的时候，交换左右指针，且left++。为0的时候就right++。
时间复杂度o(n),空间复杂度o(1)
```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int size = nums.size();
        int left = 0 , right = 0;
        while(right < size) {
            if(nums[right]) {
                swap(nums[left],nums[right]);
                left++;
            }
            right++;
        }
    }
};
```

# 到所有数组中消失的数字
给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。

找到所有在 [1, n] 范围之间没有出现在数组中的数字。

您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。

示例:

输入:
[4,3,2,7,8,2,3,1]

输出:
[5,6]

算法：哈希表，存在与否的题可考虑，

```cpp
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        //时间复杂度为n , 空间复杂度为n
        int len = nums.size();
        vector<int> ans;
        unordered_map<int,int>  myMap;
        for(int i = 0;i < len ;i++) {
            if(myMap.find(nums[i]) == myMap.end()) {
                myMap[nums[i]] = 1;
            }
        }
        for(int j = 1;j <= len; j++) {
            if(myMap.find(j) == myMap.end()){
                ans.push_back(j);
            }
        }
        return ans;
    }
};
```

# 二叉树的直径

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    int ans;
    int depth(TreeNode* rt){
        if (rt == NULL) {
            return 0; // 访问到空节点了，返回0
        }
        int L = depth(rt->left); // 左儿子为根的子树的深度
        int R = depth(rt->right); // 右儿子为根的子树的深度
        ans = max(ans, L + R + 1); // 计算d_node即L+R+1 并更新ans
        return max(L, R) + 1; // 返回该节点为根的子树的深度
    }
public:
    int diameterOfBinaryTree(TreeNode* root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }
};
```

# 数组中重复的数字
在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

示例 1：

输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 

算法：利用哈希表，不存在则保存，存在则返回。时间复杂度：o(n) , 空间复杂度o(n)

```cpp
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        unordered_map<int , int> myMap;
        int len = nums.size();
        for(int i = 0;i <len;i++) {
            if(myMap.find(nums[i]) == myMap.end()) {
                myMap[nums[i]] = 1;
            } else{
                return nums[i];
            }
        }
        return 0;
    }
};
```

# 数组排成最小的数（cvte笔试第一题）
输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
示例 1:

输入: [10,2]
输出: "102"
示例 2:

输入: [3,30,34,5,9]
输出: "3033459"

算法：其实就是一个排序的问题。但是怎么排序？
x + y < y + x => x < y , x应该放前面

解题1： 利用c++的内置排序函数sort()进行排序，时间复杂度为olog2(n)，快排+其他排序方法

```cpp
class Solution {
public:
    static bool cmp(string x , string y) {
        //排序的规则
        return x + y < y + x;
    }

    string minNumber(vector<int>& nums) {
        vector<string> store;
        int len = nums.size();
        for(int i = 0;i < len;i++) {
            //to_string将int转换为string ,atoi将string转换为int
            store.push_back(to_string(nums[i]));
        }
        // c++标准模板库的sort进行排序
        sort(store.begin(),store.end(),cmp);
        string res;
        for(auto i : store) {
            res += i;
        }
        return res;
    }
};
```

解题2：自己实现快排
```cpp
class Solution {
public:
    void fastSort(vector<string>& arr , int begin , int end) {
        if(begin > end) return;
        int i = begin;
        int j = end;
        string base = arr[begin];
        while(i != j) {
            while((arr[j] + base >= base + arr[j]) && i < j) j--;
            while((arr[i] + base <= base + arr[i]) && i < j) i++;
            if(i < j) {
                string tmp = arr[j];
                arr[j] = arr[i];
                arr[i]  = tmp;
            }
        }
        arr[begin] = arr[i];
        arr[i] = base;
        fastSort(arr , begin , i - 1);
        fastSort(arr , i + 1 ,  end);
    }

    string minNumber(vector<int>& nums) {
        int len = nums.size();
        vector<string> store;
        for(int i = 0;i < len;i++) {
            store.push_back(to_string(nums[i]));
        }
        fastSort(store , 0 , len - 1);
        string ans;
        for(auto i : store) {
            // cout<<i<<endl;
            ans += i;
        }
        return ans;
    }
};
```

# 替换空格
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

示例 1：

输入：s = "We are happy."
输出："We%20are%20happy."

算法1：最简便的方法：遇到空格就依次添加 " %20 " 时间复杂度o(n) , 空间复杂度o(n)
```cpp
class Solution {
public:
    string replaceSpace(string s) {
        int len = s.size() , count = 0;
        // for(int i = 0;i < len;  i++) {
        //     if(s[i] == ' ') 
        //         count++;
        // }
        string res;   
        for(int i = 0;i < len;i++) {
            if(s[i] == ' '){
                res+='%';
                res+='2';
                res+='0';
            }else {
                res += s[i];
            }
        }    
        return res;
    }
};
```
算法2：进阶方法：时间复杂度o(n) ， 空间复杂度o(1),在原字符串上修改
```cpp
class Solution {
public:
    string replaceSpace(string s) {
        int len = s.size() , count = 0;
        for(int i = 0;i < len;  i++) {
            if(s[i] == ' ') 
                count++;
        }
        s.resize(len + 2*count);//新的字符串的长度，因为原来的空格占一个字符，所以是 *2
        //i = j的时候i和j指向最开始，因为j每次跳过了两个空格
        for(int i = len - 1 , j = s.size() - 1;i < j;i--,j--) {
            if(s[i] != ' ')
                s[j] = s[i];
            else {
                s[j - 2] = '%';
                s[j - 1] = '2';
                s[j] = '0';
                j -=2;
            }
        }
        return s;
    }
};
```

# 倒序打印链表
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
示例 1：
输入：head = [1,3,2]
输出：[2,3,1]

算法:辅助栈后进先出。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        stack<int>  myStk;
        for(ListNode* cur = head; cur != nullptr;cur = cur->next) {
            myStk.push(cur->val);
        }
        int size = myStk.size();
        vector<int> ans;
        ans.reserve(size);
        for(int i = 0;i < size;i++) {
            ans.push_back(myStk.top());
            myStk.pop();
        }
        return ans;
    }
};
```

# 斐波那契和青蛙跳台阶
```js
/**
 * @param {number} n
 * @return {number}
 */

var numWays = function(n) {
    // 相当于斐波那契数列：滚动数组的方式
    //时间复杂度o(n) , 空间复杂度o(n)
    // let dp = [];
    // dp[1] = 1;
    // dp[0] = 1;
    // if(n === 1 || n === 0) {
    //     return dp[n];
    // }
    // for(let i = 2;i <= n;i++) {
    //     dp[i] = dp[i - 1] + dp[i - 2];
    //     dp[i] %= 1000000007;
    // }
    // return dp[n];
    //优化:空间复杂度降低为o(1)
    let sum = 1 , a = 1 , b = 1;
    if(n === 0 || n === 1) return sum;
    for(let i = 2; i <= n; i++) {
        sum = (a + b) % 1000000007;
        a = b; //b为后面一位 ， 存到a里面
        b = sum;//sum存到b中，sum下轮继续更新
    }
    return sum;
};
```

# 寻找旋转数组中的最小值(无重复数字)
假设按照升序排序的数组在预先未知的某个点上进行了旋转。例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] 。

请找出其中最小的元素。

示例 1：

输入：nums = [3,4,5,1,2]
输出：1
示例 2：

输入：nums = [4,5,6,7,0,1,2]
输出：0
示例 3：

输入：nums = [1]
输出：1

### 算法1: 遍历整个数组，当前一个比后一个大的时候，则出现了转折点，其后一个就是最小值。时间复杂度o(n)

```js
    for(let i = 0;i < nums.length - 1;i++) {
        if(nums[i] > nums[i+1])
            return nums[i+1];
    }
    return nums[0];//当没有旋转的时候，就返回第一个
```

### 算法2: 二分法，while中加等于号，在里面返回，不加等于号在外面返回。二分法最终都会落到两个数的区间内上，最终left和right会相等，此时加不加等于号就看在while中返回还是在while外返回。
具体题解见：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/solution/er-fen-cha-zhao-wei-shi-yao-zuo-you-bu-dui-cheng-z/

```js
/**
 * @param {number[]} nums
 * @return {number}
 */
 /**while循环中加了等于号*/
var findMin = function(nums) {
    // 比较右值
    let len = nums.length;
    let left = 0, right = len - 1;
    while(left <= right) {
        let mid = left + Math.floor((right - left)/2);//中间值始终会偏左，即存在左右两个值相等的情况，但是mid不会等于右值
        if(nums[mid] < nums[right]) {
            right = mid; //右边的区间这样收缩，是为了避免恰好中间是最小值。
        }else if(nums[mid] > nums[right]) {
            left = mid + 1;//左边的值一旦大于右边，那么这个值一定不是最小值
        } else {
            return nums[left];
        }
    }
};
 /**while循环中不加等于号*/
var findMin = function(nums) {
    // 比较右值
    let len = nums.length;
    let left = 0, right = len - 1;
    while(left < right) {
        let mid = left + Math.floor((right - left)/2);//中间值始终会偏左，即存在左右两个值相等的情况，但是mid不会等于右值
        if(nums[mid] < nums[right]) {
            right = mid; //右边的区间这样收缩，是为了避免恰好中间是最小值。
        }else if(nums[mid] > nums[right]) {
            left = mid + 1;//左边的值一旦大于右边，那么这个值一定不是最小值
        }
    }
    return nums[left];
};
```

# 寻找旋转数组中的最小值(有重复数字)
```js
/**
 * @param {number[]} nums
 * @return {number}
 */
var findMin = function(nums) {
    // 比较右值
    let len = nums.length;
    let left = 0, right = len - 1;
    while(left < right) {
        let mid = left + Math.floor((right - left)/2);//中间值始终会偏左，即存在左右两个值相等的情况，但是mid不会等于右值
        if(nums[mid] < nums[right]) {
            right = mid; //右边的区间这样收缩，是为了避免恰好中间是最小值。
        }else if(nums[mid] > nums[right]) {
            left = mid + 1;//左边的值一旦大于右边，那么这个值一定不是最小值
        } else {
            right = right - 1;//去重处理
        }
    }
    return nums[left];
};
```
### 复杂度分析 
时间复杂度：平均时间复杂度为 O(log n)，其中n是数组nums的长度。如果数组是随机生成的，那么数组中包含相同元素的概率很低，在二分查找的过程中，大部分情况都会忽略一半的区间。而在最坏情况下，如果数组中的元素完全相同，那么 while 循环就需要执行 n 次，每次忽略区间的右端点，时间复杂度为 O(n)。

空间复杂度：O(1)。

# 单词搜索
给定一个二维网格和一个单词，找出该单词是否存在于网格中。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

示例:

board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

给定 word = "ABCCED", 返回 true
给定 word = "SEE", 返回 true
给定 word = "ABCB", 返回 false

### 算法：dfs和回溯法。
dfs：深度优先搜索：在函数递归调用的过程中往下走，再在函数出栈的过程中再往回，即可实现回溯。
回溯：利用标记数组，将当前的位置标记，在回溯的过程中将其清除。

```js
/**
 * @param {character[][]} board
 * @param {string} word
 * @return {boolean}
 */
/**
 * 深度优先搜索：在函数递归调用的过程中往下走，再在函数出栈的过程中再往回，即可实现回溯
 */
var exist = function(board, word) {
    const h = board.length , w = board[0].length;//获取高和宽
    const directions = [[0,1],[0,-1],[1,0],[-1,0]];//四个方向，上下左右
    const visited = new Array(h); //声明一个标记数组，和原数组一样的大小
    //创建二维数组
    for(let i = 0;i < h;i++) {
        visited[i] = new Array(w).fill(false);
    }

    const check = (i , j ,s , k) => {
        if(board[i][j] != s[k]) { //判断当前点和当前的字符是否相等
            return false;
        } else if(k == s.length - 1) {//字符串只有一个字符，最后一个退出的边界条件
            return true;
        }
        //最开始的那个点相等,则开始判断四周的点
        visited[i][j] = true;//对应相等的的位置标记起来
        let result = false;
        for(const [dx , dy] of directions) {//遍历当前坐标的上下左右每个坐标
            let newi = i + dx , newj = j + dy;//当前的坐标移动
            //在矩阵范围之内
            if(newi >= 0 && newi < h && newj >= 0 && newj < w){ //注意w是长度，而数组是长度-1 ， h也是如此。因此不能加=号

                if(!visited[newi][newj]) {//当前位置没有被访问过
                    const flag = check(newi , newj , s , k+1);//深度优先搜索
                    if(flag) {
                        result = true;//寻找到一个匹配的即可
                        break;
                    }
                }
            }
        }
        //上下左右均没有和其相等的
        visited[i][j] = false;//将其复原，在for循环里从下一个点重新开始
        return result;
    }

    //遍历二维数组，从每个点开始验证是否存在路径
    for(let i = 0;i < h;i++) {
        for(let j = 0;j < w;j++) {
            if(check(i , j,word,0)) // 从每个点开始寻找
                return true;
        }
    }
    return false;
};
```

### 复杂度分析
时间复杂度：遍历二维数组：o(mn) ， check函数中最长是o(L * L* L),总的时间复杂度是o(m * n * L * L * L);
空间复杂度：开辟了一个标记数组：o(mn) ,还有函数的调用栈o(min(L , mn))

# 机器人的运动范围
算法：dfs和bfs+回溯

法1：BFS
```js
/**
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @return {number}
 */
var movingCount = function(m, n, k) {
    //建立一个标记数组 , 初始值全置为0
    const visited = new Array(m).fill().map(_ => new Array(n).fill(false))
    let queue = [[0,0]];//存入初始坐(0,0)）
    //获取坐标之和函数
    let getSum = (x,y) => {
        let sum = 0;
        while(x || y) {
            sum += x % 10;
            x=Math.floor(x/10);
            sum += y % 10;
            y = Math.floor(y/10); 
        }
        return sum;
    };
    let ans = 0;
    //在js中使用数组的长度,如果使用while(a)则不行，因为只有null,NaN,"",undefined，0，false转换为布尔值为false
    while(queue.length) {
        let [x,y] = queue.shift();
        let sum  = getSum(x,y);
        if(x > m - 1 || y > n - 1 || sum > k || visited[x][y]) {//这条语句可以避免越界，因为越界的坐标之后直接跳过
            continue;
        }
        ans++;
        visited[x][y] = true;
        queue.push([x,y+1] , [x+1,y]);//推入新数组
    }
    return ans;
};
```

法2：DFS
```JS
/**
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @return {number}
 */
var movingCount = function(m, n, k) {
    //定义二维标记数组
    let visited = new Array(m);
    for(let i = 0;i <= m;i++) {
        visited[i] = new Array(n).fill(false);
    }
    //定义坐标之和函数
    const getSum = (x,y) => {
        let sum = 0;
        while(x || y) {
            sum += x % 10;
            x = Math.floor(x/10);
            sum += y % 10;
            y = Math.floor(y/10);
        }
        return sum;
    }
    //递归函数
    const dfs = (x, y) => {
        //递归终止条件
        if(x > m - 1||y > n - 1||visited[x][y]||getSum(x,y)>k) {
            return 0;
        }
        //该盒子已经被标记访问过
        visited[x][y] = true;
        return  dfs(x,y+1) + dfs(x+1,y) + 1;//深度优先搜索
    } 
    return  dfs(0,0);
};
```

### 复杂度分析
时间复杂度：最坏o(mn)
空间复杂度：最坏o(mn)

# 剪绳子(1)
算法：动态规划。对于正整数n，可以拆分为两个正整数的和。其拆分后可以选择不细分：k * (n-k)，也可以选择细分：k * dp[n-k],dp是某一段分割后的最大长度。因此动态规划方程：
dp[i] = max(k属于(1,i-1))(max(k * (n-k),k * dp[n-k]))。
```js
/**
 * @param {number} n
 * @return {number}
 */
var integerBreak = function(n) {
    //dp[i] = max{}
    let dp = [];
    dp[0] = dp[1] =0;
    for(let i = 2;i <=n;i++) {//一步一步往上递增
        dp[i] = 0;//让初始值为0，方便下面的比较找到最大值
        for(let j = 1;j<=i-1;j++) {
            dp[i] = Math.max(dp[i], Math.max(j * (i-j),j*dp[i-j]));//dp[i-j]的范围是1到i-1，每一步的i都要依靠之前的每个1到i-1的dp，因此才需要一步一步往上走。
        }
    }
    return dp[n];
};
```
### 复杂度分析：
时间复杂度：o(n * n)  空间复杂度：o(n)

优化1：使用数学，应该尽可能的将绳子切割成3，其次就是2。切割最终都是1/2/3，这些不能再细分。
```js
/**
 * @param {number} n
 * @return {number}
 */
var integerBreak = function(n) {
    if(n <= 3) return n-1;
    let remainder = n%3;
    let quotient = Math.floor(n/3);
    if(remainder === 0) return Math.pow(3,quotient);//当能全部除尽
    if(remainder === 1) return Math.pow(3,quotient-1) * 4;//当为1的时候，则应该将3和1组合，因为2*2大于3*1
    if(remainder === 2) return Math.pow(3,quotient) * 2;//当为2的时候，则直接添加
};
```
### 复杂度分析：
时间复杂度：o(1)  空间复杂度：o(1)

优化2： 贪心算法：每次找到当前的一个局部最优解，最后再迭加起来。这里贪心就在于尽可能得到多的3 ，其次就是2.

```js
/**
 * @param {number} n
 * @return {number}
 */
var integerBreak = function(n) {
    let res = 1;
    if(n <= 3) return n - 1;
    if(n === 4) return 4; //将4单独列出来，因为4 - 3 = 1.而 1 * (n - 1) < n，故1不考虑。
    while(n > 4) {
        res *= 3;
        n -= 3;//当为7的时候，最终是4.
    }
    return res*n;
};
```
### 复杂度分析：
时间复杂度：o(n)  空间复杂度：o(1)

# 剪绳子(2 考虑大数)
算法：贪婪算法
```c++
class Solution {
public:
    int cuttingRope(int n) {
        if(n < 4) return n - 1;
        long res = 1;//必须是long，因为对1000000007取余的两个大数相乘不会超过long，
        //尽可能用３，即３的倍数的数全部用３的幂，４和２除外
        while(n > 4)//循环求余　O(n)复杂度
        {
            res *= 3;
            res %= 1000000007;//(a∗b)%c=((a%c∗b%c))%c，这里应该是数学，下面最终的返回那里还有一个取余
            n -= 3;
        }
        // 最后n的值只有可能是：2、3、4。而2、3、4能得到的最大乘积恰恰就是自身值
        // 因为2、3不需要再剪了（剪了反而变小）；4剪成2x2是最大的，2x2恰巧等于4
        return (int)(res * n % 1000000007);
    }
};
```

算法：数学方法,大数求余法
```java
class Solution {

    private int mod = (int)1e9 + 7;

    public int cuttingRope(int n) {
        if(n < 4){
            return n-1;
        }
        int cnt3 = n / 3;
        if(n % 3 == 0){
            return (int)pow(3, cnt3);
        } else if(n % 3 == 1){
            return (int)((pow(3, cnt3 - 1) * 4) % mod);
        } else {
            return (int)((pow(3, cnt3) * 2) % mod);
        }
    }

    private long pow(long base, int num){
        long res = 1;
        while(num > 0){
            if((num & 1) == 1){
                res *= base;
                res %= mod;
            }
            base *= base;
            base %= mod;
            num >>= 1;
        }
        return res;
    }
}
```

# 二进制中1的个数

算法：每次往右移动一位，此时高位是补0，并不是循环移位，再与1相与，即可。
```c++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int ans = 0;
        while(n!=0) {
            ans += n & 1;
            n = n>>1;
        }
        return ans;
    }
}; 
```
### 复杂度分析
时间复杂度：o(log2 n),
空间复杂度：o(1)

# int , long ,long long 的取值范围
```c++
unsigned int 0～4294967295
int -2147483648～2147483647
unsigned long 0～4294967295
long -2147483648～2147483647
long long的最大值：9223372036854775807
long long的最小值：-9223372036854775808
unsigned long long的最大值：18446744073709551615

__int64的最大值：9223372036854775807
__int64的最小值：-9223372036854775808
unsigned __int64的最大值：18446744073709551615
```

#整数反转（有溢出问题）
解决溢出：1. int转换成long型；2.在转换之前就进行判断；3.对1000000007取余 4.字符串处理
```c++  long 型确保不会溢出
class Solution {
public:
    int reverse(int x) {
        long long ans = 0;//用长整型来保存int溢出的问题
        while(x != 0) {
            int tmp = x % 10;//负数取余是负数
            ans = ans * 10 + tmp;
            x /= 10;
        }
        return (int)ans == ans?ans:0;//用int来强转long会保存后面32位
    }
};
```

```c++ 数学方法，提前判断
class Solution {
public:
    int reverse(int x) {
        int ans = 0;
        while(x != 0) {
            int tmp = x % 10;

            if(ans > 2147483647 /10 || (((ans == 2,147483647 /10) && tmp == 7))) return 0;
            if(ans < -2147483648 /10 || (((ans == -2147483648 /10) && tmp == 8))) return 0;

            ans = 10 *ans + tmp;
            x /= 10;
        }
        return ans;
    }
};
```

# 数值的整数次方 https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/
算法：利用二分法将运算降低，递归实现。
```c++
class Solution {
public:
    double quickPow(double x , long n) {//因为n = -2,147483648的时候取绝对值会溢出，所以long来保存
        if(n == 1) return x;//边界条件换成0也可以
        double half = quickPow(x , n/2);//递，不断将指数降低
        return (n % 2 == 0 ? 1:x)*half*half;//归，一步一步得到结果
    }

    double myPow(double x, int n) {
        //边界条件处理
        if(x == 1 || n == 0) return 1;
        if(n < 0) {
            x = 1/x;
            n = abs(n);
        }
        return quickPow(x , n);
    }
};
```

### 复杂度分析
时间复杂度:o(log(n))
空间复杂度:o(log(n))

# 删除链表的节点
算法：遍历所有节点，当下一个节点的值等于目标值的时候开始操作。注意头结点的特殊情况处理

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        ListNode* cur = head;
        if(cur->val == val) return head->next;
       while(cur->next != NULL) {
            if(cur->next->val == val) {
                cur->next = cur->next->next;//指针处理
                break;//记得退出
            }
            cur = cur->next;
        }
        return head;
    }
};
```

### 算法复杂度
时间复杂度：最坏的时间复杂度为o(n)
空间复杂度：o(1)

# 打印从1到最大的n位数 (待解决)


# 表示数值的字符串

算法：见代码注释。1 .去除空格 2.判断前面有没有数字 3.遇到空格的情况 4.遇到e的情况。index表示索引标志位，numeric表示每次结果的布尔值

```c++
class Solution {
private:
    //整数的格式可以用[+|-]B表示，其中B为无符号整数
    bool scanInterger(const string s , int& index) {
        //跳过+ 或者-号
        if(s[index] == '+' || s[index] == '-')
            ++index; //去除+和-
        return scanUnsignedInterger(s , index);
    }

    //判断是否是无符号整数
    bool scanUnsignedInterger(const string s , int& index) {
        int befor = index;
        while(index != s.size() && s[index] >= '0' && s[index] <= '9')
            index++;
        return index > befor;//大于就返回true , 说明是无符号的。若相等，则说明不会移动，即不是无符号
    }
public:
    bool isNumber(string s) {
        if(s.size() == 0)
            return false;
        int index = 0;//字符索引，一直要递增
        //字符串开始有空格的话，去除空格
        while(s[index] == ' ')
            ++index;
        //扫描除整数的部分,index停留在小数点前一位
        bool numeric = scanInterger(s,index); //numeric就是指的是每次的一个标志位
        //如果出现'.'，接下来是数字的小数部分
        if(s[index] == '.') {
            ++index;
            // 下面一行代码用||的原因：
            // 1. 小数可以没有整数部分，例如.123等于0.123；
            // 2. 小数点后面可以没有数字，例如233.等于233.0；
            // 3. 当然小数点前面和后面可以有数字，例如233.666
            numeric = scanUnsignedInterger(s , index) || numeric;
        }

        if(s[index] == 'e' || s[index] == 'E') {
            ++index;

            numeric = numeric && scanInterger(s , index);
        }

        while(s[index] == ' ') {
            ++index;
        }
        //index == s.size()代表指针是不是在结尾处就停下了，停下来了就代表是被其他字符阻断，则不是数字
        return numeric && index == s.size();

    }
};
```

#  调整数组顺序使奇数位于偶数前面 https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/
算法1：首尾指针 , 往前搜，遇到偶数，往后搜，遇到奇数，再交换。

```c++
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        while(left <= right) {
            if(nums[left] & 1 != 0) {
                left++;//跳过奇数
                continue;//这样会执行在外面的while循环，防止left一直递增到越界
            }
            if(nums[right] % 2 == 0) {
                right--;//跳过偶数
                continue;//这样会执行在外面的while循环，防止right一直递剪到越界
            }
            swap(nums[left] , nums[right]);
            //交换完之后其指针就是对应的奇数和偶数，因此再进一步
            right--;
            left++;
        }
        return nums;
    }
};
```

算法2：快慢指针
```c++
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int low = 0, fast = 0;
        while (fast < nums.size()) {
            if (nums[fast] & 1) {
                swap(nums[low], nums[fast]);
                low ++;
            }
            fast++;
        }
        return nums;
    }
};
```

# 链表中倒数第k个节点 https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/
算法：快慢指针。先让 fast指到位置 k ， 再一起动fast和slow。当fast指到尾的时候，slow就到倒数第k个位置。
fast = k,slow = 0;当fast = n ， slow = n - k。即倒数第k个
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode* slow = head;
        ListNode* fast = head;
        int fNode = k;
        while(fNode--) 
        {
            fast = fast->next;
        };

        while(fast != NULL) {
            fast = fast->next            slow = slow->next;
        }
        return slow;
    }
};
```

# 二叉树的镜像
算法1：深度优先
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if(root == NULL) return NULL;
        TreeNode* tmp = root->right; // 先保存下来，因为当下条语句生效的时候，root->right指向就变了。
        root->right = mirrorTree(root->left);
        root->left = mirrorTree(tmp);
        return root;
    }
};
```

算法2：BFS
```c++

```

# 树的子结构 https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/submissions/
算法：以二叉树的每个顶点当做root，进行是否相等判断。A树相当于做一个先序遍历
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //判断子树是否相等
    bool isSame(TreeNode* rA , TreeNode* rB) {
        if(rB == NULL) return true;
        if(rA == NULL) return false;
        if(rA->val != rB->val) return false;
        return isSame(rA->left , rB->left) && isSame(rA->right , rB->right);//遍历左右子树都相等
    }

    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(A == NULL || B == NULL) return false;
        //先序遍历之前的操作 
        if(isSame(A , B) == true) { 
            return true;
        }
        //先序遍历
        return isSubStructure(A->left , B)||isSubStructure(A->right , B);
    }
};
```

### 复杂度分析
时间复杂度：o(m * n) , m , n分别为节点数量
空间复杂度O(M):当树A和树B都退化为链表时，递归调用深度最大。当 NM≤N 时,遍历树A与递归判断的总递归深度为M;当 M>=N 时，最差情况为遍历至树A 叶子节点，此时总递归深度为M。

# 对称二叉树 https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/submissions/

```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:

    bool isSame(TreeNode* rootA , TreeNode* rootB) {
        if(rootA == NULL && rootB == NULL) return true;
        if(!rootA || !rootB) {
            return false;
        }
        if(rootA->val != rootB->val) return false;
        return isSame(rootA->left , rootB->right) && isSame(rootA->right,rootB->left);//判断是否对称
    } 

    bool isSymmetric(TreeNode* root) {
        if(root == NULL) return true;
        return isSame(root->left , root->right);
    }
};
```
### 算法复杂度
时间复杂度： 其中 n 为二叉树的节点数量，每次执行 recur() 可以判断一对节点是否对称，因此最多调用 n/2 次 递归 方法 
空间复杂度: 二叉树退化成链表 o(n)

# 顺时针打印矩阵
算法：上下左右四个边界，依次收缩
```js
/**
 * @param {number[][]} matrix
 * @return {number[]}
 */
var spiralOrder = function(matrix) {
    let ans = [];
    if(matrix.length === 0) return ans;
    let l = 0 , r = matrix[0].length - 1 , t = 0 , b = matrix.length - 1;
    while(1){
        for(let i = l;i <= r;i++) {
            ans.push(matrix[t][i]);
        }
        t++; //因为第一行完了，则收缩上边界
        if(t > b) break;//判断边界
        for(let i = t;i <= b;i++) {
            ans.push(matrix[i][r]);
        }
        r--;//最右边的完了，则收缩右边界
        if(r < l) break;
        for(let i = r;i >= l;i--) {
            ans.push(matrix[b][i]);
        }
        b--;//最下边的完了，则收缩下边界
        if(b < t) break;
        for(let i = b;i >= t;i--) {
            ans.push(matrix[i][l])
        }
        l++;//最左边的完了，则收缩左边界
        if(l > r) break;
    }
    return ans;
};
```

### 算法复杂度
时间复杂度：o(mn)
空间复杂度：o(1)

# 包含min函数的栈https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/
算法：本题考察的是将min函数的时间复杂度降为1 ， 因此需要两个栈，一个是数据栈，一个是最小值栈
```c++
class MinStack {
public:
    /** initialize your data structure here. */
    stack<int> data;//存储数据栈
    stack<int> help;//存储最小栈
    MinStack() {

    }
    
    void push(int x) {
        data.push(x);
        if(help.empty() || help.top() >= x) { //这里必须有个等于号， 避免了重复最小值被弹出
            help.push(x);
        }
    }
    
    void pop() {
        if(data.top() == help.top()) //当弹出的数据和辅助栈的数据相等，则弹出，相当于弹出最小值
            help.pop();
        }
        data.pop();
    }
    
    int top() {
        return data.top();
    }
    
    int min() {
        return help.top();
    }
;
```

### 复杂度分析
时间复杂度：因为栈的top,pop,push操作为o(1),因此各种操作的时间复杂度o(1)
空间复杂度：o(n)

# 栈的压入、弹出序列 
算法：
初始化： 辅助栈 stackstack ，弹出序列的索引 index;
遍历压栈序列： 各元素记为 num;
元素 numnum 入栈；
循环出栈：若 stack 的栈顶元素 == 弹出序列元素 popped[i] ，则执行出栈与 i++ ；
返回值： 若 stack 为空，则此弹出序列合法。
```C++
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> help;//利用辅助栈模拟操作
        int index = 0;//数组poped的下标指引
        for(auto n : pushed) { //依次弹入pushed中的元素
            help.push(n);//推入数字
            //下面这个循环是精髓。当顶部元素等于poped的元素的时候，就弹出辅助栈的顶部元素
            while(!help.empty() && (help.top() == popped[index])) {
                help.pop();
                index++;
            }
        }
        return help.empty();
    }
};
```
### 复杂度分析：
时间复杂度 O(N)： 其中 NN 为列表 pushedpushed 的长度；每个元素最多入栈与出栈一次，即最多共 2N 次出入栈操作。
空间复杂度 O(N) ： 辅助栈 stack最多同时存储 N 个元素


#  从上到下打印二叉树
算法：广度优先搜索
```JS
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
var levelOrder = function(root) {
    if(root == null) return [];
    let queue = [root];
    let ans = [];
    while(queue.length != 0) {
        let tmp = queue.shift();
        ans.push(tmp.val);
        if(tmp.left) queue.push(tmp.left);
        if(tmp.right) queue.push(tmp.right);
    }
    return ans;
};
```

# 复杂度分析
时间复杂度 O(N):N为二叉树的节点数量，即 BFS 需循环 N次。
空间复杂度 O(N):最差情况下，即当树为平衡二叉树时，最多有 N/2个树节点同时在queue中，使用O(N)大小的额外空间。

# 从上到下打印二叉树 II (每一层打印一行)
算法：BFS
```js
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var levelOrder = function(root) {
    if(root == null) return [];
    let ans = [];
    let queue = [root];
    while(queue.length) {
        const len = queue.length;
        let tmp = [];
        for(let i = 0;i < len;i++) {
            let tmpRoot = queue.shift();
            tmp.push(tmpRoot.val);
            if(tmpRoot.left) queue.push(tmpRoot.left);
            if(tmpRoot.right) queue.push(tmpRoot.right);
        }
        ans.push([...tmp]);//对象解构
        tmp.length = 0;
    }
    return ans;
};
```
算法：DFS
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:

    void dfs(TreeNode* root, vector<vector<int>>& res , int level) {
        if(root == NULL) return;
        if(level >= res.size()) res.emplace_back(vector<int>());//因为初始的res是没有给定多大空间的，所有只能这样动态赋值
        res[level].emplace_back(root->val);

        dfs(root->left , res , level+1);
        dfs(root->right , res , level+1);
    }

    vector<vector<int>> levelOrder(TreeNode* root) {
        if(root == NULL) return {};
        vector<vector<int>> res;
        dfs(root , res , 0);
        return res;
    }
};
```

# 从上到下打印二叉树 III(之字形打印)
算法：bfs+双端队列
```js
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var levelOrder = function(root) {
    if(root == null) return [];
    let res = [];
    let queue = [root];
    let flag = false;//奇偶标志
    while(queue.length) {
        let deque = [] , len = queue.length;
        while(len--) {
            let tmpRoot = queue.shift();
            if(!flag) 
                deque.push(tmpRoot.val);
            else
                deque.unshift(tmpRoot.val);
            if(tmpRoot.left) queue.push(tmpRoot.left);
            if(tmpRoot.right) queue.push(tmpRoot.right);
        }
        res.push([...deque]);
        deque.length = 0;
        flag = !flag;
    } 
    return res;
};
```
算法复杂度：
时间复杂度:o(N)
空间复杂度:o(N)

算法2：dfs + 插入
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:

    void dfs(TreeNode* root , vector<vector<int>>& res , int level) {
        if(root == NULL) return;
        if(res.size() <= level) res.resize(level + 1);

        if(level % 2 == 0) //不能用res.size()因为到最后res会变
            res[level].emplace_back(root->val);
        else {
            res[level].insert(res[level].begin(),root->val);
        }

        dfs(root->left , res , level+1);
        dfs(root->right , res , level+1);
    }

    vector<vector<int>> levelOrder(TreeNode* root) {
        if(root == NULL) return {};
        vector<vector<int>> res;
        dfs(root , res , 0);
        return res;
    }
};
```

算法：迭代(未曾使用双端队列)
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(root == NULL) return {};
        vector<vector<int>> res;
        queue<TreeNode*> queue;
        queue.push(root);
        bool flag = true;
        while(!queue.empty()) {
            int size = queue.size();
            vector<int> tmpRes(size);//初始化大小
            for(int i = 0;i < size;i++) {
                TreeNode* tmpRoot = queue.front();
                queue.pop();
                int index = flag?i:size - 1 - i;//i的索引来
                tmpRes[index] = tmpRoot->val;
                if(tmpRoot->left) queue.push(tmpRoot->left);
                if(tmpRoot->right) queue.push(tmpRoot->right);
            }
            flag = !flag;
            res.emplace_back(tmpRes);
        }
        return res;
    }
};
```

# 二叉搜索树的后序遍历序列 https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/(单调栈法)
算法：分治算法。根据根节点进行切割 ， 变成左右子树。再分别判断左右子树。类似于快排 
```c++
class Solution {
public:

    bool helper(vector<int> arr , int i , int j) {
        if(i >= j) return true; //当只有一个节点或者没有节点的时候，就返回true
        int m = i; //存下节点i
        while(arr[m] < arr[j]) m++;//找到右节点的第一个值
        int p = m;
        //判断右子树是否符合条件
        while(p++ < j) {
            if(arr[p] < arr[j])
                return false;
        }
        return helper(arr , i , m - 1) && helper(arr , m , j - 1);
    }

    bool verifyPostorder(vector<int>& postorder) {
        int len = postorder.size();
        return helper(postorder , 0 , len - 1);
    }
};
```


# 路径总和2 https://leetcode-cn.com/problems/path-sum-ii/
算法：DFS + 回溯 ， 递归写法
```c++
class Solution {
public:
    vector<vector<int>> ret;
    vector<int> path;

    void dfs(TreeNode* root, int sum) {
        if (root == nullptr) {
            return;
        }
        path.emplace_back(root->val);
        sum -= root->val;
        if (root->left == nullptr && root->right == nullptr && sum == 0) {
            ret.emplace_back(path);
        }
        dfs(root->left, sum);
        dfs(root->right, sum);
        path.pop_back();
    }

    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        dfs(root, sum);
        return ret;
    }
};
```

###  复杂度分析
时间复杂度：O(N^2) , 
空间复杂度：O(N)


# 复杂链表的复制

算法1：利用哈希表，存储下原来的链表的位置和新开辟的链表的位置。首先，先赋值简单的链表，存储下对应关系，再赋值随机链表
```C++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        Node* copyHead = new Node(0);
        Node* p = copyHead;
        unordered_map<Node* , Node*> help;

        Node* hd = head;
        //深拷贝常规节点
        while(hd != nullptr) {
            p->next = new Node(hd->val);
            help[hd] = p->next;
            p = p->next;
            hd = hd->next;
        }
        hd = head;//指回
        p = copyHead->next;
        while(hd != nullptr) {
            p->random = help[hd->random];//因为在help中存下了hd和p的对应关系，两个是互不相关的，在根据hd->random的指向节点，就可以找到其对应节点
            hd = hd->next;
            p = p->next;
        }
        return copyHead->next;
    }
};
```

### 算法复杂度分析：
时间复杂度：o(N)
空间复杂度：o(N)

待优化？？？？？

















