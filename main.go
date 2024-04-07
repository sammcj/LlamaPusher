package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"

	"github.com/sergi/go-diff/diffmatchpatch"
)

const (
	ollamaURL     = "http://localhost:11434/api/generate"
	regenerateMsg = "♻️ Regenerate Commit Messages"
	contentType   = "application/json"
)

var (
	typeToGitmoji = map[string]string{
		"feat":     "✨",
		"fix":      "🚑",
		"docs":     "📝",
		"style":    "💄",
		"refactor": "♻️",
		"test":     "✅",
		"chore":    "🔧",
	}
)

type OllamaRequest struct {
	Model             string `json:"model"`
	Prompt            string `json:"prompt"`
	Stream            bool   `json:"stream"`
	MaxTokens         int    `json:"max_tokens"`
	TopP              int    `json:"top_p"`
	Temperature       int    `json:"temperature"`
	RepetitionPenalty int    `json:"repetition_penalty"`
}

type OllamaResponse struct {
	Response string `json:"response"`
}

func main() {
	model             := flag.String("model", "tinydolphin:1.1b-v2.8-q5_K_M", "The model to use for generating commit messages")
	language          := flag.String("language", "english", "The language to use for generating commit messages")
	template          := flag.String("template", "", "The template to use for formatting commit messages")
	emoji             := flag.Bool("emoji", true, "Add gitmoji to the commit message")
	commitType        := flag.String("commit-type", "", "The type of commit (e.g., feat, fix, docs)")
	list              := flag.Bool("list", false, "Generate a list of commit message options")
	force             := flag.Bool("force", false, "Force the commit without prompting for confirmation")
	filterFee         := flag.Bool("filter-fee", false, "Display the approximate fee for using the API")
	maxTokens         := flag.Int("max-tokens", 2048, "The maximum number of tokens to generate")
	topP              := flag.Int("top-p", 1, "The top-p sampling value")
	temperature       := flag.Int("temperature", 1, "The temperature value for sampling")
	repetitionPenalty := flag.Int("repetition-penalty", 1, "The repetition penalty value")
	filterFiles				:= flag.String("filter-files", "", "Filter files to commit (glob pattern)")
	flag.Parse()

	fmt.Printf("AI provider: ollama, Model: %s\n", *model)

	if !checkGitRepository() {
		log.Fatal("This is not a git repository 🙅‍♂️")
	}

	diff := getGitDiff(*filterFiles)
	if diff == "" {
		fmt.Println("No changes to commit 🙅")
		fmt.Println("Maybe you forgot to add the files? Try git add . and then run this script again.")
		os.Exit(1)
	}

	if *list {
		err := generateListCommits(diff, *model, *language, *template, *emoji, *commitType, *maxTokens, *topP, *temperature, *repetitionPenalty, *filterFee)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		err := generateSingleCommit(diff, *model, *language, *template, *emoji, *commitType, *maxTokens, *topP, *temperature, *repetitionPenalty, *force, *filterFee, *filterFiles)
		if err != nil {
			log.Fatal(err)
		}
	}
}

func checkGitRepository() bool {
	cmd := exec.Command("git", "rev-parse", "--is-inside-work-tree")
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	return strings.TrimSpace(string(output)) == "true"
}

func getGitDiff(filterFiles string) string {
    cmd := exec.Command("git", "diff", "--staged" , "--no-color", "--no-prefix")
		if filterFiles != "" {
			cmd.Args = append(cmd.Args, filterFiles)
		}
    output, err := cmd.Output()
    if err != nil {
        log.Fatal(err)
    }

    dmp := diffmatchpatch.New()
    diffs := dmp.DiffMain(string(output), "", true)

    var diffLines []string
    for _, diff := range diffs {
        if diff.Type == diffmatchpatch.DiffEqual {
            continue
        }

        lines := strings.Split(diff.Text, "\n")
        for _, line := range lines {
            if strings.HasPrefix(line, "@@") || strings.HasPrefix(line, "diff --git") {
                continue
            }
            diffLines = append(diffLines, line)
        }
    }

    return strings.Join(diffLines, "\n")
}

func generateSingleCommit(diff, model, language, template string, doAddEmoji bool, commitType string, maxTokens, topP, temperature, repetitionPenalty int, force, filterFee bool, filterFiles string) error {
    diff = getGitDiff(filterFiles)

    if diff == "" {
        fmt.Println("No changes to commit 🙅")
        fmt.Println("Maybe you forgot to add the files? Try git add . and then run this script again.")
        os.Exit(1)
    }

    prompt := getPromptForSingleCommit(diff, commitType, language)

	proceed, err := filterAPI(prompt, 1, maxTokens, filterFee)
	if err != nil {
		return err
	}
	if !proceed {
		os.Exit(1)
	}

	text, err := sendMessageOllama(prompt, model, maxTokens, topP, temperature, repetitionPenalty)
	if err != nil {
		return err
	}

	finalCommitMessage := text
	if doAddEmoji {
		finalCommitMessage = addGitmojiToCommitMessage(finalCommitMessage)
	}

	if template != "" {
		finalCommitMessage = processTemplate(template, finalCommitMessage)
		fmt.Printf("Proposed Commit With Template:\n------------------------------\n%s\n------------------------------\n", finalCommitMessage)
	} else {
		fmt.Printf("Proposed Commit:\n------------------------------\n%s\n------------------------------\n", finalCommitMessage)
	}

	if force {
		makeCommit(finalCommitMessage)
		return nil
	}

	fmt.Print("Do you want to continue? (y/n): ")
	reader := bufio.NewReader(os.Stdin)
	answer, _ := reader.ReadString('\n')
	answer = strings.TrimSpace(answer)
	if strings.ToLower(answer) != "y" {
		fmt.Println("Commit aborted by user 🙅‍♂️")
		os.Exit(1)
	}

	makeCommit(finalCommitMessage)
	return nil
}

func generateListCommits(diff, model, language, template string, doAddEmoji bool, commitType string, maxTokens, topP, temperature, repetitionPenalty int, filterFee bool) error {
	numOptions := 5
	prompt := getPromptForListCommits(diff, commitType, language, numOptions)

	proceed, err := filterAPI(prompt, numOptions, maxTokens, filterFee)
	if err != nil {
		return err
	}
	if !proceed {
		os.Exit(1)
	}

	text, err := sendMessageOllama(prompt, model, maxTokens, topP, temperature, repetitionPenalty)
	if err != nil {
		return err
	}

	msgs := strings.Split(text, ";")
	for i := range msgs {
		msgs[i] = strings.TrimSpace(msgs[i])
		if doAddEmoji {
			msgs[i] = addGitmojiToCommitMessage(msgs[i])
		}
		if template != "" {
			msgs[i] = processTemplate(template, msgs[i])
		}
	}

	msgs = append(msgs, regenerateMsg)

	var selectedMsg string
	fmt.Println("Select a commit message:")
	for i, msg := range msgs {
		fmt.Printf("%d. %s\n", i+1, msg)
	}
	fmt.Print("Enter your choice (1-6): ")
	var choice int
	fmt.Scanln(&choice)

	if choice < 1 || choice > len(msgs) {
		fmt.Println("Invalid choice. Exiting.")
		os.Exit(1)
	}

	selectedMsg = msgs[choice-1]
	if selectedMsg == regenerateMsg {
		return generateListCommits(diff, model, language, template, doAddEmoji, commitType, maxTokens, topP, temperature, repetitionPenalty, filterFee)
	}

	makeCommit(selectedMsg)
	return nil
}

func getPromptForSingleCommit(diff, commitType, language string) string {
	prompt := "From the following git diff create a short, useful git commit message in " + language + " language"

	if commitType != "" {
		prompt += " with commit type '" + commitType + "'. "
	} else {
		prompt += ". "
	}

	prompt += "Do not preface the commit with anything, use the present tense, return the full sentence, " +
		"and use the conventional commits specification (<type in lowercase>: <subject>): " +
		"START OF GIT DIFF:\n" +
		diff +
		"\nEND OF GIT DIFF"

	return prompt
}

func getPromptForListCommits(diff, commitType, language string, numOptions int) string {
	prompt := "From the following git diff create a short, useful git commit message in " + language + " language"

	if commitType != "" {
		prompt += " with commit type '" + commitType + "', "
	} else {
		prompt += ", "
	}

	prompt += "and make " + fmt.Sprint(numOptions) + " options that are separated by ';'. " +
		"For each option, use the present tense, return the full sentence, " +
		"and use the conventional commits specification (<type in lowercase>: <subject>): " +
		"START OF GIT DIFF:\n" +
		diff +
		"\nEND OF GIT DIFF"

	return prompt
}

func processTemplate(template, commitMessage string) string {
	finalCommitMessage := strings.ReplaceAll(template, "{COMMIT_MESSAGE}", commitMessage)

	if strings.Contains(finalCommitMessage, "{GIT_BRANCH}") {
		cmd := exec.Command("git", "branch", "--show-current")
		output, err := cmd.Output()
		if err != nil {
			log.Fatal(err)
		}
		currentBranch := strings.TrimSpace(string(output))
		finalCommitMessage = strings.ReplaceAll(finalCommitMessage, "{GIT_BRANCH}", currentBranch)
	}

	return finalCommitMessage
}

func sendMessageOllama(prompt, model string, maxTokens, topP, temperature, repetitionPenalty int) (string, error) {
	data := OllamaRequest{
		Model:             model,
		Prompt:            prompt,
		Stream:            false,
		MaxTokens:         maxTokens,
		TopP:              topP,
		Temperature:       temperature,
		RepetitionPenalty: repetitionPenalty,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return "", err
	}

	resp, err := http.Post(ollamaURL, contentType, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var ollamaResp OllamaResponse
	err = json.NewDecoder(resp.Body).Decode(&ollamaResp)
	if err != nil {
		return "", err
	}

	return ollamaResp.Response, nil
}

func makeCommit(commitMessage string) {
	fmt.Println("Committing Message... 🚀")
	cmd := exec.Command("git", "commit", "-m", commitMessage)
	err := cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Commit Successful! 🎉")
}

func filterAPI(prompt string, numCompletion, maxTokens int, filterFee bool) (bool, error) {
	numTokens := len(strings.Fields(prompt))
	fee := float64(numTokens)/1000*0.02 + (0.001 * float64(numCompletion))

	if numTokens > maxTokens {
		fmt.Printf("The commit diff is too large. Max %d tokens allowed.\n", maxTokens)
		return false, nil
	}

	if filterFee {
		fmt.Printf("This will cost you ~$%.3f for using the API.\n", fee)
		fmt.Print("Do you want to continue 💸? (y/n): ")
		reader := bufio.NewReader(os.Stdin)
		answer, _ := reader.ReadString('\n')
		answer = strings.TrimSpace(answer)
		if strings.ToLower(answer) != "y" {
			return false, nil
		}
	}

	return true, nil
}

func addGitmojiToCommitMessage(commitMessage string) string {
	re := regexp.MustCompile(`\b[a-zA-Z]+\b`)
	match := re.FindString(commitMessage)

	if match == "" {
		return commitMessage
	}

	if gitmoji, ok := typeToGitmoji[match]; ok {
		return gitmoji + " " + commitMessage
	}

	return commitMessage
}
