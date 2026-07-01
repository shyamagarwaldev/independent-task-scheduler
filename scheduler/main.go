package main

import (
	"os"

	"k8s.io/component-base/cli"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"

	"github.com/shyamagarwaldev/independent-task-scheduler/scheduler/pkg/plugin"
)

func main() {
	command := app.NewSchedulerCommand(
		app.WithPlugin(plugin.Name, plugin.New),
	)
	code := cli.Run(command)
	os.Exit(code)
}
